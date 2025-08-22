#!/usr/bin/env python3
"""
Simple test script to verify Qdrant connection and basic functionality.
Run this before using the main insertion script.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import openai

# Load environment variables from the correct location
load_dotenv("/home/chorizo/projects/benjamin_ai_system/agentic_app/config/.env")
load_dotenv()  # Also load from current directory if exists

def test_qdrant_connection():
    """Test connection to Qdrant."""
    print("Testing Qdrant connection...")
    
    QDRANT_URL = os.getenv("AGENTIC_APP_QDRANT_URL", "http://127.0.0.1:6333")
    QDRANT_API_KEY = os.getenv("AGENTIC_APP_QDRANT_API_KEY")
    
    # Parse Qdrant URL
    from urllib.parse import urlparse
    parsed_url = urlparse(QDRANT_URL)
    QDRANT_HOST = parsed_url.hostname or "localhost"
    QDRANT_PORT = parsed_url.port or 6333
    
    try:
        if QDRANT_API_KEY:
            client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                api_key=QDRANT_API_KEY
            )
        else:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Test basic connection
        collections = client.get_collections()
        print(f"✅ Qdrant connection successful!")
        print(f"   URL: {QDRANT_URL}")
        print(f"   Existing collections: {len(collections.collections)}")
        
        # Check for the specific collection we need
        benjamin_memory_exists = False
        for collection in collections.collections:
            print(f"   - {collection.name}")
            if collection.name == "benjamin_agent_memory":
                benjamin_memory_exists = True
                try:
                    collection_info = client.get_collection("benjamin_agent_memory")
                    print(f"     └─ Points: {collection_info.points_count}")
                except:
                    pass
        
        if benjamin_memory_exists:
            print("✅ Required collection 'benjamin_agent_memory' found!")
        else:
            print("⚠️  Collection 'benjamin_agent_memory' not found - you may need to create it first")
        
        return client
        
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return None

def test_openai_api():
    """Test OpenAI API connection."""
    print("\nTesting OpenAI API...")
    
    OPENAI_API_KEY = os.getenv("AGENTIC_APP_OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("AGENTIC_APP_EMBEDDING_MODEL", "text-embedding-3-small")
    
    if not OPENAI_API_KEY:
        print("❌ AGENTIC_APP_OPENAI_API_KEY not found in environment")
        return False
    
    try:
        openai.api_key = OPENAI_API_KEY
        
        # Test with a simple embedding
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input="Test embedding"
        )
        
        embedding = response.data[0].embedding
        print(f"✅ OpenAI API connection successful!")
        print(f"   Model: {EMBEDDING_MODEL}")
        print(f"   Embedding dimension: {len(embedding)}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Benjamin AI System - Vector Store Connection Test")
    print("=" * 50)
    
    # Test Qdrant
    qdrant_client = test_qdrant_connection()
    
    # Test OpenAI
    openai_ok = test_openai_api()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if qdrant_client and openai_ok:
        print("✅ All tests passed! Ready to run the main insertion script.")
    else:
        print("❌ Some tests failed. Please check your configuration.")
        if not qdrant_client:
            print("   - Fix Qdrant connection")
        if not openai_ok:
            print("   - Fix OpenAI API configuration")

if __name__ == "__main__":
    main()
