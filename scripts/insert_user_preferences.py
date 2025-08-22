#!/usr/bin/env python3
"""
Script to insert user preferences, insights, and health patterns into Qdrant vector store
for the Benjamin AI System.

This script:
1. Connects to Qdrant vector database
2. Creates embeddings for text data using OpenAI
3. Inserts user preferences, insights, and health patterns with metadata
4. Creates collections for different data types
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any
import asyncio

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    PointStruct,
    CreateCollection,
    UpdateCollection,
    CollectionInfo
)
import openai
from dotenv import load_dotenv

# Load environment variables from the correct location
load_dotenv("/home/chorizo/projects/benjamin_ai_system/agentic_app/config/.env")
load_dotenv()  # Also load from current directory if exists

# Configuration - using the same env vars as the agentic app
QDRANT_URL = os.getenv("AGENTIC_APP_QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.getenv("AGENTIC_APP_QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("AGENTIC_APP_OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("AGENTIC_APP_EMBEDDING_MODEL", "text-embedding-3-small")

# Parse Qdrant URL
from urllib.parse import urlparse
parsed_url = urlparse(QDRANT_URL)
QDRANT_HOST = parsed_url.hostname or "localhost"
QDRANT_PORT = parsed_url.port or 6333

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# User data to insert
USER_DATA = {
    "preferences": [
        "Wants to manage food cravings and develop a healthier relationship with food.",
        "Seeks to strengthen motivation for a balanced sports routine.",
        "Aims to reach a target weight of 83 kg.",
        "Desires a structured coaching plan with regular follow-up over several months.",
        "Wants to eat based on hunger, not emotional triggers like fatigue or loneliness.",
        "Wishes to feel physically ready for the arrival of a future baby.",
        "Aspires to feel comfortable and confident in his clothes.",
        "Prefers a nutritionist with a focus on endurance sports and physical performance.",
        "Interested in a flexitarian diet and recipes.",
        "Believes a coach should use HRV, RHR, and sleep data to determine workouts.",
        "Interested in learning how to manage eating disorders and food cravings.",
        "Looking for books on nutrition for endurance sports based on scientific meta-analyses.",
        "Values a data-driven approach to coaching, using metrics like VO2max and training load.",
        "Hopes to regain the physical fitness to enjoy running and cycling freely."
    ],
    "insights": [
        "At 41, august 2025, and has a history of significant weight fluctuation, from 105 kg down to 82 kg.",
        "Previous attempts at weight loss have been successful in the short term but not sustainable.",
        "Feels most satisfied with his physical state when he is running consistently, a realization he often has in hindsight.",
        "Has had mixed results with hypnosis; it was very effective for emotional processing but ineffective for long-term weight loss.",
        "Believes his eating habits are deeply ingrained and require a tailored, long-term approach to change.",
        "Identifies his weight as a primary obstacle to improving his running performance.",
        "At 41, august 2025, weighs 103kg at a height of 1m78.",
        "Trains 3-4 times per week, including running, cycling on a gravel bike, and strength training.",
        "His primary motivations are preparing for fatherhood and achieving personal fitness and aesthetic goals."
    ],
    "health_patterns": [
        "Eating is often triggered by being alone, tired, bored, or frustrated.",
        "Experiences periods of obsession with junk food.",
        "When cooking alone, tends to prepare unhealthy 'comfort' foods in excessive quantities.",
        "Has a habit of eating very large portions, both alone and with his partner.",
        "Tends to eat too quickly.",
        "Uses post-workout fatigue as a justification for overeating.",
        "Avoids running long distances due to a fear of injury, likely related to his weight.",
        "Expresses a desire to better manage food cravings and potential disordered eating patterns.",
        "Recognizes that loneliness, frustration, and guilt are key emotional drivers of his eating behaviors."
    ]
}

class QdrantUserDataManager:
    """Manages user data insertion into Qdrant vector store."""
    
    def __init__(self):
        """Initialize Qdrant client and OpenAI."""
        # Initialize Qdrant client
        if QDRANT_API_KEY:
            self.client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                api_key=QDRANT_API_KEY
            )
        else:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        self.embedding_model = EMBEDDING_MODEL
        self.vector_size = 1536  # Dimension for text-embedding-3-small
        
        # Use existing collection
        self.collection_name = "benjamin_agent_memory"
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for given text using OpenAI."""
        try:
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding for text: {text[:50]}...")
            print(f"Error: {e}")
            raise
    
    def ensure_collection_exists(self):
        """Check if the collection exists."""
        try:
            # Check if collection exists
            collection_info = self.client.get_collection(self.collection_name)
            print(f"✅ Collection '{self.collection_name}' already exists")
            print(f"   Points count: {collection_info.points_count}")
            return True
        except Exception as e:
            print(f"❌ Collection '{self.collection_name}' not found: {e}")
            print(f"Please create the collection first or check your connection.")
            return False
    
    def insert_data_points(self, data_type: str, data_items: List[str]):
        """Insert data points into the collection."""
        points = []
        
        # Map data types to match existing collection format
        type_mapping = {
            "preferences": "user_preference",
            "insights": "user_insight", 
            "health_patterns": "health_pattern"
        }
        
        mapped_type = type_mapping.get(data_type, data_type)
        
        for idx, item in enumerate(data_items):
            try:
                # Create embedding
                embedding = self.create_embedding(item)
                
                # Create metadata matching existing collection structure
                metadata = {
                    "text": item,
                    "type": mapped_type,
                    "source": "user_profile_import",
                    "user_id": "benjamin",
                    "inserted_at": datetime.utcnow().isoformat(),
                    "index": idx
                }
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=metadata
                )
                points.append(point)
                
                print(f"Prepared point for {mapped_type}[{idx}]: {item[:60]}...")
                
            except Exception as e:
                print(f"Error processing item {idx} in {data_type}: {e}")
                continue
        
        # Insert points in batch
        if points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"Successfully inserted {len(points)} points into '{self.collection_name}'")
            except Exception as e:
                print(f"Error inserting points into '{self.collection_name}': {e}")
                raise
        else:
            print(f"No valid points to insert into '{self.collection_name}'")
    
    def insert_all_user_data(self):
        """Insert all user data into the collection."""
        print("Starting user data insertion into Qdrant...")
        
        # Check collection exists
        if not self.ensure_collection_exists():
            return False
        
        # Insert each data type
        for data_type, data_items in USER_DATA.items():
            print(f"\nInserting {len(data_items)} {data_type} items...")
            self.insert_data_points(data_type, data_items)
        
        print("\nUser data insertion completed!")
        return True
    
    def verify_insertions(self):
        """Verify that data was inserted correctly."""
        print("\nVerifying insertions...")
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            total_points = collection_info.points_count
            
            # Count points by type to verify our insertions
            total_expected = sum(len(items) for items in USER_DATA.values())
            
            print(f"Collection '{self.collection_name}':")
            print(f"  Total points: {total_points}")
            print(f"  Expected new points: {total_expected}")
            
            # Check each type
            type_mapping = {
                "preferences": "user_preference",
                "insights": "user_insight", 
                "health_patterns": "health_pattern"
            }
            
            for data_type, mapped_type in type_mapping.items():
                expected_count = len(USER_DATA[data_type])
                print(f"  Expected {mapped_type}: {expected_count} items")
            
            print("✅ Verification completed - check Qdrant web UI for detailed view")
                    
        except Exception as e:
            print(f"❌ Error verifying collection: {e}")
    
    def search_example(self, query: str, limit: int = 3):
        """Example search to test the vector store."""
        print(f"\nTesting search for: '{query}' in '{self.collection_name}'")
        
        try:
            # Create embedding for query
            query_embedding = self.create_embedding(query)
            
            # Search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                score = result.score
                text = result.payload.get("text", "No text")
                data_type = result.payload.get("type", "unknown")
                source = result.payload.get("source", "unknown")
                print(f"  {i}. [Score: {score:.3f}] [{data_type}] {text[:60]}...")
                print(f"      Source: {source}")
                
        except Exception as e:
            print(f"Error during search: {e}")


def main():
    """Main function to run the insertion script."""
    print("Benjamin AI System - User Data Vector Store Insertion")
    print("=" * 60)
    
    # Check required environment variables
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        return
    
    print(f"Qdrant connection: {QDRANT_URL}")
    print(f"OpenAI embedding model: {EMBEDDING_MODEL}")
    print("")
    
    try:
        # Initialize manager
        manager = QdrantUserDataManager()
        
        # Insert data
        success = manager.insert_all_user_data()
        
        if not success:
            print("❌ Failed to insert data - check your Qdrant connection")
            return
        
        # Verify insertions
        manager.verify_insertions()
        
        # Test with example searches
        print("\n" + "=" * 60)
        print("TESTING SEARCHES")
        print("=" * 60)
        
        # Test searches
        test_queries = [
            "weight loss motivation",
            "eating behavior patterns", 
            "running performance",
            "endurance sports nutrition"
        ]
        
        for query in test_queries:
            manager.search_example(query)
        
        print("\n✅ Script completed successfully!")
        
    except Exception as e:
        print(f"❌ Script failed: {e}")
        raise


if __name__ == "__main__":
    main()
