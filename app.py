import streamlit as st
from multi_agent import run_multi_agent

st.set_page_config(
    page_title="Multi-Agent Medical System",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .header-container {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .agent-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .triage { background-color: #fff3e0; color: #e65100; }
    .scheduling { background-color: #e3f2fd; color: #1565c0; }
    .eligibility { background-color: #f3e5f5; color: #6a1b9a; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 style="color:white; margin:0">🏥 Multi-Agent Medical System</h1>
    <p style="color:#c8e6c9; margin:0">Orchestrated AI agents for triage, scheduling, and eligibility</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🤖 Agent Architecture")
    st.markdown("""
    **Supervisor** routes queries to:
    
    🔴 **Triage Agent**
    - Assesses urgency
    - Determines specialty
    
    🔵 **Scheduling Agent**  
    - Finds providers
    - Books appointments
    - Processes referrals
    
    🟣 **Eligibility Agent**
    - Checks insurance
    - Verifies eligibility
    - Prior authorization
    """)
    st.markdown("---")
    st.markdown("### Try asking:")
    st.markdown("""
    - *"P001 has chest pain, book urgent cardiology"*
    - *"Check P002 insurance then schedule dermatology"*
    - *"Process referral for P003: knee pain, needs orthopedics"*
    - *"Does P001 need prior auth for an MRI?"*
    """)
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"],
        avatar="🏥" if message["role"] == "assistant" else "👤"):
        st.write(message["content"])

# Chat input
query = st.chat_input("Describe the patient situation...")

if query:
    with st.chat_message("user", avatar="👤"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("Agents collaborating..."):
            answer = run_multi_agent(query)
        st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })