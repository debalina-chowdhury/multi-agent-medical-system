import os
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "test")

def test_required_packages():
    """Test all required packages are installed"""
    import anthropic
    import langgraph
    import chromadb
    assert anthropic is not None
    assert langgraph is not None
    assert chromadb is not None

def test_knowledge_base_creates():
    """Test knowledge base initializes"""
    from knowledge_base import create_knowledge_base
    retriever = create_knowledge_base()
    assert retriever is not None

def test_knowledge_base_retrieves():
    """Test RAG retrieval returns results"""
    from knowledge_base import retrieve_policies
    result = retrieve_policies("prior authorization MRI")
    assert len(result) > 0

def test_find_provider_tool():
    """Test find provider tool"""
    from agents import find_provider
    result = find_provider.invoke({"specialty": "cardiology"})
    assert "DR00" in result

def test_check_insurance_tool():
    """Test insurance check tool"""
    from agents import check_insurance
    result = check_insurance.invoke({"patient_id": "P001"})
    assert "active" in result.lower()