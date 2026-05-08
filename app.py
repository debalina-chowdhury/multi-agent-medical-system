import streamlit as st
from multi_agent import run_multi_agent

st.set_page_config(
    page_title="Multi-Agent Medical System",
    page_icon="🏥",
    layout="centered"
)

st.markdown("""
<style>
    .header-container {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-container">
    <h1 style="color:white; margin:0">🏥 Multi-Agent Medical System</h1>
    <p style="color:#c8e6c9; margin:0">Supervisor · Triage · Scheduling · Eligibility · RAG Knowledge Base</p>
</div>
""", unsafe_allow_html=True)

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
    
    📚 **RAG Knowledge Base**
    - Medical policies
    - Insurance guidelines
    - Scheduling rules
    """)
    st.markdown("---")
    st.markdown("### Try asking:")
    st.markdown("""
    - *"P001 has chest pain, book urgent cardiology"*
    - *"Does P002 need prior auth for an MRI?"*
    - *"What's the policy for new patients unseen 2 years?"*
    - *"Process referral for P003: knee pain, orthopedics"*
    """)
    if st.button("🗑️ Clear"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"],
        avatar="🏥" if message["role"] == "assistant" else "👤"):
        st.write(message["content"])

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