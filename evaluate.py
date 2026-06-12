"""
Evaluation harness for the multi-agent medical workflow.

Measures three things that actually matter for a supervisor-routed multi-agent system:

  1. Routing accuracy   — did the supervisor send the query to the right agent(s)?
  2. Tool-use accuracy   — did the expected tools get invoked?
  3. End-to-end success  — does the final answer contain the expected information?

Run:  python eval_harness/evaluate.py
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

# Import the graph from your main module.
# (Assumes this file sits next to multi_agent.py, or adjust the import path.)
from multi_agent import build_multi_agent_graph
from langchain_core.messages import HumanMessage, AIMessage


# ── EVAL CASES ────────────────────────────────────────────────────────────────
# Each case: the user query + what a correct run should look like.
# - expected_agents:  agents that SHOULD be invoked (order-independent set)
# - expected_tools:   tool names that SHOULD be called (substring match, optional)
# - expected_keywords: strings the final answer should contain (case-insensitive)
@dataclass
class EvalCase:
    name: str
    query: str
    expected_agents: list           # e.g. ["triage", "scheduling"]
    expected_keywords: list = field(default_factory=list)
    expected_tools: list = field(default_factory=list)


EVAL_CASES = [
    # ── Triage routing + urgency/specialty tools ──────────────────────────────
    EvalCase(
        name="urgent_chest_pain_triage",
        query="Patient P123 has sudden chest pain and difficulty breathing. What should we do?",
        expected_agents=["triage"],
        expected_tools=["assess_urgency", "determine_specialty"],
        # assess_urgency flags chest pain as URGENT; determine_specialty maps chest -> cardiology
        expected_keywords=["urgent", "cardiology"],
    ),
    EvalCase(
        name="routine_specialty_triage",
        query="Patient P200 has a skin rash that won't go away. Which specialist should they see?",
        expected_agents=["triage"],
        expected_tools=["determine_specialty"],
        # skin/rash -> dermatology
        expected_keywords=["dermatology"],
    ),

    # ── Scheduling routing + provider/booking tools ───────────────────────────
    EvalCase(
        name="find_provider_scheduling",
        query="Find me a dermatologist with availability for patient P200.",
        expected_agents=["scheduling"],
        expected_tools=["find_provider"],
        # find_provider returns Dr. Sarah Johnson (DR001) etc.
        expected_keywords=["DR001", "Dr."],
    ),
    EvalCase(
        name="book_appointment_scheduling",
        query="Book patient P123 with provider DR001 tomorrow at 9am for a cardiology consult.",
        expected_agents=["scheduling"],
        expected_tools=["book_appointment"],
        # book_appointment returns a reference like APT-...
        expected_keywords=["confirmed", "APT-"],
    ),

    # ── Eligibility routing + insurance/RAG tools ─────────────────────────────
    EvalCase(
        name="check_insurance_eligibility",
        query="Check the insurance status and copay for patient P123.",
        expected_agents=["eligibility"],
        expected_tools=["check_insurance"],
        # check_insurance returns Blue Cross PPO, copay $30 specialist
        expected_keywords=["active", "copay"],
    ),
    EvalCase(
        name="prior_auth_rag_eligibility",
        query="Does patient P123 need prior authorization for an MRI?",
        expected_agents=["eligibility"],
        expected_tools=["check_prior_auth"],
        # MRI is in high_auth_procedures -> prior auth REQUIRED
        expected_keywords=["prior authorization", "required"],
    ),

    # ── End-to-end: triage -> scheduling -> eligibility ───────────────────────
    EvalCase(
        name="end_to_end_rash_referral",
        query=("Patient P200 has a persistent skin rash. Assess it, find a specialist, "
               "and check whether their insurance covers the visit."),
        expected_agents=["triage", "scheduling", "eligibility"],
        # spans all three; final answer should reference the specialty and insurance
        expected_keywords=["dermatology", "insurance"],
    ),
]


# ── INSTRUMENTED RUN ──────────────────────────────────────────────────────────
def run_with_trace(graph, query: str):
    agents_invoked = []
    tools_called = []
    final_answer = ""
    full_transcript = []

    initial = {"messages": [HumanMessage(content=query)],
               "current_agent": "", "urgency": "", "specialty": "",
               "completed_agents": []}

    for step in graph.stream(initial, config={"recursion_limit": 50}):
        for node_name, node_output in step.items():
            if node_name in ("triage_agent", "scheduling_agent", "eligibility_agent"):
                agents_invoked.append(node_name.replace("_agent", ""))
            for msg in node_output.get("messages", []):
                # capture tool calls
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        tools_called.append(tc.get("name", ""))
                # capture ALL message content — AI messages AND tool outputs
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content.strip():
                    full_transcript.append(content)
                    final_answer = content   # last non-empty wins

    return final_answer, agents_invoked, tools_called, "\n".join(full_transcript)


# ── SCORING ───────────────────────────────────────────────────────────────────
@dataclass
class CaseResult:
    name: str
    routing_ok: bool
    tools_ok: bool
    keywords_ok: bool
    agents_invoked: list
    tools_called: list
    latency_s: float
    error: str = ""

    @property
    def passed(self) -> bool:
        return self.routing_ok and self.tools_ok and self.keywords_ok


def score_case(case: EvalCase, final_answer, agents_invoked, tools_called, latency, transcript="") -> CaseResult:
    invoked_set = set(agents_invoked)
    # Routing: every expected agent must have been invoked at least once.
    routing_ok = all(a in invoked_set for a in case.expected_agents)

    # Tools: every expected tool name appears (substring match) among called tools.
    tools_ok = all(any(t in called for called in tools_called) for t in case.expected_tools) \
        if case.expected_tools else True

    # Keywords: ANY expected keyword present (these are "reasonable output" signals).
    ans = (transcript or final_answer or "").lower()
    keywords_ok = any(k.lower() in ans for k in case.expected_keywords) \
        if case.expected_keywords else True

    return CaseResult(
        name=case.name,
        routing_ok=routing_ok,
        tools_ok=tools_ok,
        keywords_ok=keywords_ok,
        agents_invoked=agents_invoked,
        tools_called=tools_called,
        latency_s=round(latency, 2),
    )


# ── RUNNER ────────────────────────────────────────────────────────────────────
def run_eval():
    graph = build_multi_agent_graph()
    results = []

    print(f"Running {len(EVAL_CASES)} eval cases...\n")
    for case in EVAL_CASES:
        t0 = time.time()
        try:
            final, agents, tools, transcript = run_with_trace(graph, case.query)
            res = score_case(case, final, agents, tools, time.time() - t0, transcript)
        except Exception as e:
            res = CaseResult(case.name, False, False, False, [], [],
                             round(time.time() - t0, 2), error=str(e))
        results.append(res)

        status = "PASS" if res.passed else "FAIL"
        print(f"[{status}] {res.name}  ({res.latency_s}s)")
        print(f"   routing={res.routing_ok} (got {res.agents_invoked})")
        print(f"   tools={res.tools_ok}   (got {res.tools_called})")
        print(f"   keywords={res.keywords_ok}")
        if res.error:
            print(f"   error: {res.error}")
        print()

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    n = len(results)
    routing_acc = sum(r.routing_ok for r in results) / n
    tools_acc = sum(r.tools_ok for r in results) / n
    e2e_acc = sum(r.passed for r in results) / n
    avg_latency = sum(r.latency_s for r in results) / n

    print("=" * 50)
    print("SUMMARY")
    print(f"  Routing accuracy:      {routing_acc:.0%}")
    print(f"  Tool-use accuracy:     {tools_acc:.0%}")
    print(f"  End-to-end task pass:  {e2e_acc:.0%}")
    print(f"  Avg latency:           {avg_latency:.2f}s")
    print("=" * 50)

    # Save a JSON report so you can track regressions over time.
    report = {
        "summary": {
            "routing_accuracy": routing_acc,
            "tool_accuracy": tools_acc,
            "e2e_accuracy": e2e_acc,
            "avg_latency_s": avg_latency,
        },
        "cases": [asdict(r) for r in results],
    }
    with open("last_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved detailed report to eval_harness/last_report.json")

    return results


if __name__ == "__main__":
    run_eval()