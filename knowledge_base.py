from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Medical knowledge base
medical_docs = [
    # Insurance policies
    "Cardiology referrals require prior authorization for stress tests, echocardiograms, and cardiac catheterization procedures under most PPO plans.",
    "Dermatology appointments for new patients require a referral from primary care physician for insurance coverage under HMO plans.",
    "Orthopedic consultations for knee and joint pain typically require X-rays before MRI authorization is granted.",
    "Mental health appointments are covered at 80% after deductible under Blue Cross PPO plans. No referral required.",
    "Prior authorization for MRI scans takes 2-3 business days and requires clinical notes from the referring physician.",
    "Emergency cardiology appointments can bypass prior authorization requirements when patient presents with acute symptoms.",
    "Physical therapy requires prior authorization after 6 sessions under most insurance plans.",
    
    # Scheduling policies
    "Patients unseen for 18 months or more are classified as new patients and must be scheduled in new patient appointment slots.",
    "New patient appointments are typically 60 minutes. Follow-up appointments are 30 minutes.",
    "Urgent care appointments for chest pain, difficulty breathing, or severe symptoms should be same-day or next-day.",
    "Specialist appointments require referral from primary care physician for HMO plan members.",
    "Telehealth appointments are available for follow-up visits and non-urgent consultations.",
    
    # Provider policies  
    "Dr. Sarah Johnson (DR001) specializes in cardiology and sees new patients on Monday and Wednesday mornings.",
    "Dr. Michael Chen (DR002) specializes in general cardiology and interventional cardiology.",
    "New patient appointments with specialists require insurance verification at least 24 hours before the appointment.",
    "Patients must bring photo ID, insurance card, and completed new patient forms to their first appointment.",
    
    # Referral policies
    "Referral processing time is 24-48 hours for routine referrals and same-day for urgent referrals.",
    "Referrals must include: patient demographics, diagnosis code (ICD-10), reason for referral, and referring physician NPI.",
    "Electronic referrals are processed faster than fax referrals. Average fax processing time is 4-6 hours.",
    "Prior authorization for surgery requires clinical documentation, failed conservative treatment history, and imaging results.",
]

def create_knowledge_base():
    """Create and return the medical knowledge base retriever."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    docs = text_splitter.create_documents(medical_docs)
    
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="medical_policies"
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 2})

# Global retriever instance
retriever = create_knowledge_base()

def retrieve_policies(query: str) -> str:
    """Retrieve relevant medical policies for a query."""
    docs = retriever.invoke(query)
    if not docs:
        return "No specific policy found for this query."
    results = "\n".join([f"- {doc.page_content}" for doc in docs])
    return f"Relevant policies:\n{results}"