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

def test_execute_tool_find_provider():
    """Test find provider tool"""
    from agents import execute_tool
    result = execute_tool("find_provider", {"specialty": "cardiology"})
    assert "DR00" in result

def test_execute_tool_check_insurance():
    """Test insurance check tool"""
    from agents import execute_tool
    result = execute_tool("check_insurance", {"patient_id": "P001"})
    assert "active" in result.lower()