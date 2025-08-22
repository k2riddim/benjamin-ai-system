# Benjamin AI System - Vector Store Management

This directory contains scripts for managing the Qdrant vector store used by the Benjamin AI System for storing and retrieving user preferences, insights, and health patterns.

## Scripts

### `insert_user_preferences.py`
Main script to insert user preferences, insights, and health patterns into the Qdrant vector store.

**Features:**
- Creates embeddings using OpenAI's text-embedding-3-small model
- Organizes data into separate collections for different data types
- Includes metadata for each vector point
- Provides verification and testing functionality
- Handles batch insertion for efficiency

**Collection Used:**
- `benjamin_agent_memory` - Existing collection for all agent memory data including user preferences, insights, and health patterns

## Setup

1. **Install Dependencies:**
   ```bash
   pip install qdrant-client openai python-dotenv
   ```

2. **Configure Environment:**
   ```bash
   cp config_example.env .env
   # Edit .env with your actual API keys and configuration
   ```

3. **Start Qdrant (if running locally):**
   ```bash
   # Using Docker
   docker run -p 6333:6333 qdrant/qdrant
   
   # Or using Docker Compose (recommended)
   # Add qdrant service to your docker-compose.yml
   ```

## Usage

### Basic Insertion
```bash
cd scripts
python insert_user_preferences.py
```

### What the script does:
1. Connects to Qdrant vector database
2. Verifies the existing `benjamin_agent_memory` collection exists
3. Generates embeddings for each text item using OpenAI
4. Inserts vectors with metadata matching existing collection structure
5. Verifies insertion success
6. Runs example searches to test functionality

### Output Example:
```
Benjamin AI System - User Data Vector Store Insertion
============================================================
Qdrant connection: localhost:6333
OpenAI model: text-embedding-3-small

Creating collection 'benjamin_user_preferences'...
Collection 'benjamin_user_preferences' created successfully
...
Successfully inserted 14 points into 'benjamin_user_preferences'
...
✅ Script completed successfully!
```

## Data Structure

Each vector point includes:
- **Vector**: 1536-dimensional embedding from OpenAI
- **Metadata** (matching existing collection structure):
  - `text`: Original text content
  - `type`: Mapped type (user_preference, user_insight, health_pattern)
  - `source`: "user_profile_import" (identifies this data source)
  - `user_id`: "benjamin" (static for this system)
  - `inserted_at`: ISO timestamp
  - `index`: Original index in the data array

## Integration with AI System

The vector store enables:
- **Semantic search** for relevant user information
- **Context retrieval** for agent conversations
- **Personalized recommendations** based on user preferences
- **Pattern recognition** in user behavior and health data

## Troubleshooting

### Common Issues:

1. **Connection Error to Qdrant:**
   - Ensure Qdrant is running on the specified host/port
   - Check firewall settings
   - Verify QDRANT_HOST and QDRANT_PORT in .env

2. **OpenAI API Issues:**
   - Verify OPENAI_API_KEY is set correctly
   - Check API quota and billing
   - Ensure internet connectivity

3. **Memory Issues:**
   - Large datasets may require batch processing
   - Consider using smaller embedding models for testing

### Verification Commands:
```python
# Check collection status
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
print(client.get_collections())

# Count points in collection
collection_info = client.get_collection("benjamin_agent_memory")
print(f"Points: {collection_info.points_count}")

# Search for specific types
results = client.search(
    collection_name="benjamin_agent_memory",
    query_filter={"must": [{"key": "source", "match": {"value": "user_profile_import"}}]},
    limit=10
)
```

## Future Enhancements

- **Incremental Updates**: Add functionality to update existing vectors
- **Data Versioning**: Track changes in user preferences over time
- **Multi-user Support**: Extend to support multiple users
- **Advanced Search**: Add filtering and complex query capabilities
- **Performance Monitoring**: Add metrics and logging
