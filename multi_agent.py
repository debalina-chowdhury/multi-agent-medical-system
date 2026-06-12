import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from agents import (
    triage_llm, scheduling_llm, eligibility_llm,
    triage_tools, scheduling_tools, eligibility_tools
)

load_dotenv()

# ── STATE ─────────────────────────────────────────────────────────────────────
class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str
    urgency: str
    specialty: str
    completed_agents: list

# ── SUPERVISOR ────────────────────────────────────────────────────────────────
supervisor_llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0
)


def supervisor(state: MultiAgentState) -> dict:
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    completed = state.get("completed_agents", [])

    system = """You are a medical workflow supervisor orchestrating a multi-step task.
    Decide which agent handles the NEXT step, considering what's ALREADY been done.

    Available agents:
    - triage: assess symptoms, urgency, determine specialty
    - scheduling: find providers, book appointments, process referrals
    - eligibility: check insurance, verify eligibility, prior authorization
    - end: the user's FULL request has been satisfied

    IMPORTANT: A complex request may need MULTIPLE agents in sequence. Do not end
    until every part of the user's original request is addressed. For example, if the
    user asked to assess symptoms AND find a specialist AND check insurance, you must
    run triage, then scheduling, then eligibility — only choose 'end' once all the
    requested parts are done.

    Respond with ONLY one word: triage, scheduling, eligibility, or end."""

    # Reconstruct the user's original request (first human message) so the
    # supervisor reasons about the WHOLE task, not just the last reply.
    original_request = ""
    for m in messages:
        if isinstance(m, HumanMessage):
            original_request = m.content
            break

    response = supervisor_llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=(
            f"User's original request: {original_request}\n"
            f"Agents already completed: {list(set(completed)) or 'none'}\n"
            f"Most recent message: {last_message}\n"
            f"Which agent should handle the next step? (or 'end' if the full request is satisfied)"
        ))
    ])

    decision = response.content.strip().lower()
    if decision not in ["triage", "scheduling", "eligibility", "end"]:
        decision = "end"

    return {"current_agent": decision}



def route_supervisor(state: MultiAgentState) -> Literal["triage_agent", "scheduling_agent", "eligibility_agent", "__end__"]:
    """Routes to the correct agent based on supervisor decision."""
    agent = state.get("current_agent", "end")
    if agent == "triage":
        return "triage_agent"
    elif agent == "scheduling":
        return "scheduling_agent"
    elif agent == "eligibility":
        return "eligibility_agent"
    return "__end__"



# ── SPECIALIZED AGENTS ────────────────────────────────────────────────────────
def triage_agent(state: MultiAgentState) -> dict:
    """Assesses urgency and determines specialty needed."""
    print("🔴 TRIAGE AGENT called") 
    system = """You are a medical triage specialist. 
    Assess patient symptoms for urgency and determine the right specialty.
    Use your tools to evaluate the situation, then summarize findings clearly.
    After completing triage, your response will be routed to scheduling or eligibility."""

    messages = [SystemMessage(content=system)] + state["messages"]
    response = triage_llm.invoke(messages)
    completed = state.get("completed_agents", [])
    return {"messages": [response], "completed_agents": completed + ["triage"]}

def scheduling_agent(state: MultiAgentState) -> dict:
    """Handles provider lookup, appointment booking, and referrals."""
    system = """You are a medical scheduling specialist.
    Find providers, book appointments, and process referrals.
    Always use find_provider first — never ask users for internal provider IDs.
    Confirm all bookings clearly with reference numbers."""
    print("🔵 SCHEDULING AGENT called")
    messages = [SystemMessage(content=system)] + state["messages"]
    response = scheduling_llm.invoke(messages)
    completed = state.get("completed_agents", [])
    return {"messages": [response], "completed_agents": completed + ["scheduling"]}

# def eligibility_agent(state: MultiAgentState) -> dict:
#     """Handles insurance verification and prior authorization."""
#     system = """You are a medical insurance eligibility specialist.
#     Check insurance status, verify eligibility, and handle prior authorization.
#     Be clear about coverage, copays, and any authorization requirements."""

#     messages = [SystemMessage(content=system)] + state["messages"]
#     response = eligibility_llm.invoke(messages)
#     return {"messages": [response]}


def eligibility_agent(state: MultiAgentState) -> dict:
    print("🟣 ELIGIBILITY AGENT called")
    system = """You are a medical insurance eligibility specialist.
    
    When asked about insurance coverage or eligibility:
    - ALWAYS call check_insurance first to get current status
    - ALWAYS call verify_eligibility if a provider is mentioned
    - ALWAYS call retrieve_medical_policy to check relevant policies
    - Do NOT ask the user if they want you to verify — just do it
    - Give a complete answer with all relevant details
    
    Never ask permission to use your tools. Use them proactively."""
    
    messages = [SystemMessage(content=system)] + state["messages"]
    response = eligibility_llm.invoke(messages)
    completed = state.get("completed_agents", [])
    return {"messages": [response], "completed_agents": completed + ["eligibility"]}

# ── TOOL ROUTING ──────────────────────────────────────────────────────────────
def route_tools(state: MultiAgentState) -> Literal["triage_tools", "scheduling_tools", "eligibility_tools", "supervisor"]:
    """Routes to correct tool node or back to supervisor."""
    messages = state["messages"]
    if not messages:
        return "supervisor"

    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        current = state.get("current_agent", "")
        if current == "triage":
            return "triage_tools"
        elif current == "scheduling":
            return "scheduling_tools"
        elif current == "eligibility":
            return "eligibility_tools"

    return "supervisor"

# ── BUILD GRAPH ───────────────────────────────────────────────────────────────
def build_multi_agent_graph():
    graph = StateGraph(MultiAgentState)

    # Add nodes
    graph.add_node("supervisor", supervisor)
    graph.add_node("triage_agent", triage_agent)
    graph.add_node("scheduling_agent", scheduling_agent)
    graph.add_node("eligibility_agent", eligibility_agent)
    graph.add_node("triage_tools", ToolNode(triage_tools))
    graph.add_node("scheduling_tools", ToolNode(scheduling_tools))
    graph.add_node("eligibility_tools", ToolNode(eligibility_tools))

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor routes to agents
    graph.add_conditional_edges("supervisor", route_supervisor)

    # Agents route to tools or back to supervisor
    graph.add_conditional_edges("triage_agent", route_tools)
    graph.add_conditional_edges("scheduling_agent", route_tools)
    graph.add_conditional_edges("eligibility_agent", route_tools)

    # Tools always return to their agent
    graph.add_edge("triage_tools", "triage_agent")
    graph.add_edge("scheduling_tools", "scheduling_agent")
    graph.add_edge("eligibility_tools", "eligibility_agent")

    return graph.compile()

multi_agent_graph = build_multi_agent_graph()


def run_multi_agent(user_query: str) -> str:
    result = multi_agent_graph.invoke(
        {
            "messages": [HumanMessage(content=user_query)],
            "current_agent": "",
            "urgency": "",
            "specialty": "",
            "completed_agents": []
        },
        config={"recursion_limit": 50}
    )
    messages = result.get("messages", [])
    if not messages:
        return "I was unable to process that request. Please try again."
    
    # Find last AI message with actual text content
    for message in reversed(messages):
        if hasattr(message, 'content') and isinstance(message.content, str) and message.content.strip():
            return message.content
    
    return "I was unable to process that request. Please try again."