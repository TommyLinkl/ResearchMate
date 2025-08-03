#!/usr/bin/env python3
"""
Test script for Enhanced RAG Pipeline
This script demonstrates the Phase 2 enhanced features
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_system_status():
    """Test basic system functionality"""
    print("🔍 Testing System Status...")
    
    # Health check
    response = requests.get(f"{API_BASE}/health")
    health = response.json()
    print(f"   Health: {health['status']}")
    print(f"   LLM Available: {health['llm_available']}")
    print(f"   Embedding Initialized: {health['embedding_initialized']}")
    
    # Get stats
    response = requests.get(f"{API_BASE}/api/stats")
    stats = response.json()
    print(f"   Pipeline Version: {stats['pipeline_version']}")
    print(f"   Enhanced RAG: {stats['enhanced_rag_enabled']}")
    print(f"   Total Documents: {stats['total_documents']}")
    print(f"   Embedding Model: {stats['embedding_model']}")
    print()

def test_enhanced_rag_config():
    """Test enhanced RAG configuration"""
    print("⚙️  Testing Enhanced RAG Configuration...")
    
    # Get current config
    response = requests.get(f"{API_BASE}/api/enhanced-rag/config")
    config = response.json()
    print(f"   Query Expansion: {config['query_expansion_enabled']}")
    print(f"   Reranking: {config['reranking_enabled']}")
    print(f"   Features: {len(config['features'])} available")
    print()

def simulate_file_upload():
    """Simulate a successful file upload to test document processing"""
    print("📄 Testing Document Upload Simulation...")
    
    # Create a test text file content
    test_content = """
    Machine Learning Interatomic Potentials
    
    Machine learning (ML) interatomic potentials represent a revolutionary approach to molecular dynamics simulations.
    These potentials combine the accuracy of quantum mechanical calculations with the efficiency of classical force fields.
    
    Key advantages include:
    1. High accuracy comparable to density functional theory (DFT)
    2. Computational efficiency for large-scale simulations  
    3. Transferability across different chemical environments
    4. Ability to capture complex many-body interactions
    
    The development process typically involves:
    - Training on high-quality quantum mechanical data
    - Feature engineering using atomic descriptors
    - Model architecture design (neural networks, Gaussian processes, etc.)
    - Validation on diverse chemical systems
    
    Applications span materials science, chemistry, and physics, enabling studies of:
    - Phase transitions and thermodynamic properties
    - Mechanical properties and defect behavior
    - Surface reactions and catalysis
    - Ion transport in batteries and fuel cells
    
    Recent advances focus on universal potentials that can handle multiple chemical elements
    and diverse bonding environments with a single trained model.
    """
    
    # Save as temporary file and upload
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file_path = f.name
    
    try:
        # Upload the file
        with open(temp_file_path, 'rb') as f:
            files = {'files': ('ml_potentials.txt', f, 'text/plain')}
            response = requests.post(f"{API_BASE}/api/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Upload successful: {result['files_processed']} files processed")
            print(f"   Total documents: {result['total_documents']}")
            print(f"   Enhanced RAG ready: {result.get('enhanced_rag_ready', False)}")
            return True
        else:
            print(f"   ❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    
    finally:
        # Clean up
        os.unlink(temp_file_path)
    
    print()

def test_enhanced_query(query, description):
    """Test enhanced RAG query with detailed analysis"""
    print(f"🤖 Testing Query: {description}")
    print(f"   Query: '{query}'")
    
    # Test with enhanced RAG enabled
    payload = {
        "query": query,
        "llm_model": "Llama-3.1-8B",
        "use_retrieval": True,
        "use_enhanced_rag": True,
        "enable_query_expansion": True,
        "enable_reranking": True,
        "use_chain_of_thought": True
    }
    
    start_time = time.time()
    response = requests.post(f"{API_BASE}/api/query", json=payload)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"   ✅ Query successful ({end_time - start_time:.2f}s)")
        print(f"   Enhanced RAG used: {result.get('enhanced_rag_used', False)}")
        print(f"   Query expanded: {result.get('query_expanded', False)}")
        print(f"   Documents retrieved: {result.get('retrieved_docs_count', 'N/A')}")
        print(f"   Reranked: {result.get('reranked', False)}")
        print(f"   Sources: {len(result.get('sources', []))}")
        
        # Show first part of answer
        answer = result.get('answer', 'No answer')
        preview = answer[:200] + "..." if len(answer) > 200 else answer
        print(f"   Answer preview: {preview}")
        
        return True
    else:
        print(f"   ❌ Query failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False
    
    print()

def main():
    """Run all tests"""
    print("🚀 ResearchMate Enhanced RAG Test Suite")
    print("=" * 50)
    
    # Test 1: System status
    test_system_status()
    
    # Test 2: Configuration
    test_enhanced_rag_config()
    
    # Test 3: Upload a test document
    upload_success = simulate_file_upload()
    
    if upload_success:
        # Test 4: Enhanced queries
        test_queries = [
            ("What are machine learning interatomic potentials?", "Basic conceptual query"),
            ("How do ML potentials compare to DFT calculations?", "Comparison query"),
            ("What are the applications of ML potentials in materials science?", "Application-focused query"),
            ("Explain the training process for ML potentials", "Process explanation query")
        ]
        
        for query, description in test_queries:
            test_enhanced_query(query, description)
            time.sleep(1)  # Small delay between queries
    
    else:
        print("⚠️  Skipping query tests due to upload failure")
        print("\n💡 To fix upload issues:")
        print("   1. Check embedding model compatibility")
        print("   2. Verify API keys if using OpenAI embeddings")
        print("   3. Try restarting the server after installing tf-keras")
    
    print("\n🏁 Test suite complete!")
    print("\nNext steps to see the enhanced RAG in action:")
    print("1. Upload your PDF file via the web interface")
    print("2. Ask questions about the content")
    print("3. Check the response metadata for enhanced features")

if __name__ == "__main__":
    main()