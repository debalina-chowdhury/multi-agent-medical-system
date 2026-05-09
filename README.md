[![CI](https://github.com/debalina-chowdhury/multi-agent-medical-system/actions/workflows/python-app.yml/badge.svg)](https://github.com/debalina-chowdhury/multi-agent-medical-system/actions)
# Multi-Agent Medical System with RAG

A production-grade multi-agent AI system built with LangGraph, Anthropic Claude, ChromaDB, and Streamlit. Uses a supervisor-pattern architecture where a central orchestrator routes queries to specialized agents, each with their own tools and a shared RAG knowledge base.

## Architecture

User Query
    │
    ▼
Supervisor Agent (Claude)
    │
    ├──► Triage Agent ──► assess_urgency, determine_specialty
    │
    ├──► Scheduling Agent ──► find_provider, book_appointment, process_referral
    │
    └──► Eligibility Agent ──► check_insurance, verify_eligibility,
                               check_prior_auth, retrieve_medical_policy (RAG)
                                        │
                                        ▼
                               ChromaDB Vector Store
                               (Medical Policy Knowledge Base)

## Key Concepts Demonstrated

- Multi-agent orchestration with LangGraph supervisor pattern
- RAG (Retrieval Augmented Generation) using ChromaDB and sentence-transformers
- Conditional edge routing between specialized agents
- ToolNode execution with automatic tool selection
- TypedDict state management across agent transitions
- Prompt engineering for specialized agent behavior

## Agents

- Supervisor: Routes queries to correct agent using Claude reasoning
- Triage Agent: Assesses urgency and determines specialty needed
- Scheduling Agent: Finds providers, books appointments, processes referrals
- Eligibility Agent: Checks insurance, verifies eligibility, handles prior authorization with RAG

## RAG Knowledge Base

The eligibility agent retrieves relevant medical policies from a ChromaDB vector store before answering coverage questions. This grounds responses in actual policy documents rather than relying on LLM general knowledge.

Knowledge base includes:
- Insurance coverage policies
- Prior authorization requirements
- Scheduling rules and new patient policies
- Provider availability and referral workflows

## Example Queries

Single agent:
- Check insurance for patient P001
- Find an orthopedic surgeon for patient P003
- Does P002 need prior authorization for an MRI?

Multi-agent coordination:
- Patient P001 has severe chest pain — assess urgency, check insurance, and book urgent cardiology today
- Process referral for P003: chronic knee pain, needs orthopedic evaluation, check if prior auth needed
- What is the policy for patients unseen for over 18 months?

## Tech Stack

- LangGraph StateGraph with conditional edge routing
- Anthropic Claude (claude-sonnet-4-5) via LangChain Anthropic
- ChromaDB vector store for RAG
- Sentence Transformers (all-MiniLM-L6-v2) for embeddings
- Streamlit web interface
- Python 3.9

## Setup

1. Clone the repo
2. Install dependencies: pip install -r requirements.txt
3. Add API key: echo "ANTHROPIC_API_KEY=your-key-here" > .env
4. Run: streamlit run app.py
5. Open http://localhost:8501

## Project Structure

- agents.py - Tool definitions for all specialized agents
- knowledge_base.py - ChromaDB vector store and RAG retriever
- multi_agent.py - LangGraph StateGraph, supervisor, and agent nodes
- app.py - Streamlit frontend
- requirements.txt - Dependencies
- .env - API key not pushed to GitHub
- .gitignore - Ignores .env

## Why This Architecture

Traditional single-agent systems with many tools become unreliable as complexity grows. The supervisor pattern solves this by giving each agent focused responsibility, letting the supervisor make routing decisions based on conversation context, and making the system easier to debug and extend.

## Future Improvements

- Add LangGraph checkpointing for conversation persistence
- Connect to real EHR APIs (Epic, eClinicalWorks, Athena)
- Integrate real insurance eligibility APIs (Availity, Change Healthcare)
- Add a fourth agent for billing and claims processing
- Deploy with authentication for multi-user support

## Author

Debalina Chowdhury
github.com/debalina-chowdhury
