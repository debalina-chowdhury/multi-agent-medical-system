import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from knowledge_base import retrieve_policies

load_dotenv()

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0
)

# ── TRIAGE TOOLS ─────────────────────────────────────────────────────────────
@tool
def assess_urgency(symptoms: str, patient_id: str) -> str:
    """Assess the urgency level of a patient's symptoms."""
    urgent_keywords = ["chest pain", "difficulty breathing", "stroke", "severe", "urgent", "emergency"]
    is_urgent = any(keyword in symptoms.lower() for keyword in urgent_keywords)
    if is_urgent:
        return f"Patient {patient_id} — URGENT. Symptoms: {symptoms}. Recommend same-day or next-day appointment. Flag for priority scheduling."
    return f"Patient {patient_id} — ROUTINE. Symptoms: {symptoms}. Standard scheduling applies."

@tool
def determine_specialty(symptoms: str) -> str:
    """Determine which medical specialty is needed based on symptoms."""
    specialty_map = {
        "chest": "cardiology",
        "heart": "cardiology",
        "skin": "dermatology",
        "rash": "dermatology",
        "bone": "orthopedics",
        "joint": "orthopedics",
        "knee": "orthopedics",
        "eye": "ophthalmology",
        "vision": "ophthalmology",
        "mental": "psychiatry",
        "anxiety": "psychiatry",
        "depression": "psychiatry",
    }
    symptoms_lower = symptoms.lower()
    for keyword, specialty in specialty_map.items():
        if keyword in symptoms_lower:
            return f"Recommended specialty: {specialty}"
    return "Recommended specialty: primary care (general practitioner)"

# ── SCHEDULING TOOLS ──────────────────────────────────────────────────────────
@tool
def find_provider(specialty: str, urgent: bool = False) -> str:
    """Find available providers by specialty."""
    if urgent:
        return (
            f"URGENT slots for {specialty}: "
            f"Dr. Sarah Johnson (DR001) — today 3pm or tomorrow 9am. "
            f"Dr. Michael Chen (DR002) — today 5pm or tomorrow 11am."
        )
    return (
        f"Available providers for {specialty}: "
        f"Dr. Sarah Johnson (DR001) — Mon 9am, Wed 2pm, Fri 10am. "
        f"Dr. Michael Chen (DR002) — Tue 11am, Thu 3pm, Fri 2pm."
    )

@tool
def book_appointment(
    patient_id: str,
    provider_id: str,
    appointment_time: str,
    reason: str = "General consultation"
) -> str:
    """Book an appointment for a patient."""
    return (
        f"Appointment confirmed for patient {patient_id} with {provider_id} "
        f"at {appointment_time} for {reason}. "
        f"SMS confirmation sent. Reference: APT-{patient_id}-{provider_id[:4].upper()}"
    )

@tool
def process_referral(referral_text: str, patient_id: str = "unknown") -> str:
    """Process an incoming referral document."""
    return (
        f"Referral processed for patient {patient_id}. "
        f"Summary: {referral_text[:100]}... "
        f"Action: specialist consult needed, insurance pre-auth required, follow-up within 48hrs."
    )

# ── ELIGIBILITY TOOLS WITH RAG ────────────────────────────────────────────────
@tool
def check_insurance(patient_id: str) -> str:
    """Check insurance status for a patient."""
    return (
        f"Patient {patient_id} insurance: ACTIVE. "
        f"Plan: Blue Cross PPO. "
        f"Expires: Dec 2026. "
        f"Copay: $30 specialist, $15 primary care. "
        f"Deductible: $500 remaining."
    )

@tool
def verify_eligibility(patient_id: str, provider_id: str) -> str:
    """Verify if patient is eligible to see a specific provider."""
    return (
        f"Patient {patient_id} is eligible for provider {provider_id}. "
        f"In-network: YES. No referral required. Pre-authorization: not needed."
    )

@tool
def check_prior_auth(patient_id: str, procedure: str) -> str:
    """Check if prior authorization is needed using medical policy knowledge base."""
    # RAG — retrieve relevant policies first
    policy_context = retrieve_policies(f"prior authorization {procedure}")
    
    high_auth_procedures = ["MRI", "CT scan", "surgery", "specialist", "stress test", "echocardiogram"]
    needs_auth = any(p.lower() in procedure.lower() for p in high_auth_procedures)
    
    if needs_auth:
        return (
            f"Prior authorization REQUIRED for {procedure} for patient {patient_id}. "
            f"Estimated approval: 2-3 business days. "
            f"Policy context: {policy_context}"
        )
    return (
        f"No prior authorization needed for {procedure} for patient {patient_id}. "
        f"Policy context: {policy_context}"
    )

@tool
def retrieve_medical_policy(query: str) -> str:
    """Retrieve relevant medical policies and insurance guidelines from knowledge base."""
    return retrieve_policies(query)

# Tool groups per agent
triage_tools = [assess_urgency, determine_specialty]
scheduling_tools = [find_provider, book_appointment, process_referral]
eligibility_tools = [check_insurance, verify_eligibility, check_prior_auth, retrieve_medical_policy]

# Bind tools to models
triage_llm = llm.bind_tools(triage_tools)
scheduling_llm = llm.bind_tools(scheduling_tools)
eligibility_llm = llm.bind_tools(eligibility_tools)