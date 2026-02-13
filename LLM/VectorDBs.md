# Vector Databases: Quick Overview

## **What is a Vector Database?**

A **vector database** is a specialized database designed to store, index, and search high-dimensional vectors (embeddings) efficiently.

**Think of it like this:**
- Traditional database: Stores exact data (names, numbers, dates) → searches for exact matches
- Vector database: Stores semantic meaning as numbers → searches for similar meaning

**Example:**
```python
# Traditional DB
"Find customers WHERE name = 'John Smith'"  # Exact match

# Vector DB  
"Find documents similar to 'machine learning tutorials'"  # Semantic similarity
# Matches: "AI learning guides", "ML education resources", etc.
```

---

## **Why Do We Need Vector Databases?**

**Problem:** Regular databases can't handle semantic search efficiently.

```python
# What you want to search:
query = "affordable Italian restaurants nearby"

# Traditional keyword search would miss:
- "Budget-friendly pasta places in the area"  ❌
- "Cheap trattorias close by"                 ❌
- "Inexpensive Mediterranean dining local"    ❌

# Vector search finds all of these ✅
# Because they have similar MEANING, even with different words
```

**Use cases:**
- 🔍 **Semantic Search**: Find similar documents, images, products
- 🤖 **RAG (Retrieval Augmented Generation)**: Give LLMs relevant context
- 🎵 **Recommendation Systems**: "Users who liked X also liked Y"
- 📸 **Image/Video Search**: Find visually similar content
- 🔐 **Fraud Detection**: Find similar transaction patterns
- 💬 **Chatbots**: Retrieve relevant knowledge

---

## **How Vector Databases Work**

### **The 4-Step Process:**

```
1. EMBED: Convert data → vectors
   "The cat sat"  →  [0.2, 0.8, 0.3, ...]
   
2. INDEX: Organize vectors for fast search
   Build ANN (Approximate Nearest Neighbor) index
   
3. STORE: Save vectors + metadata
   Vector + original text + metadata
   
4. SEARCH: Find similar vectors
   Query → embed → find nearest neighbors
```

---

## **Step-by-Step: General Workflow**

### **Step 1: Embedding (Convert Data to Vectors)**

Transform your data into numerical vectors that capture semantic meaning.

```python
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert text to vectors
texts = [
    "The quick brown fox",
    "A fast auburn canine",
    "Python is a programming language"
]

embeddings = model.encode(texts)

print(embeddings.shape)
# Output: (3, 384)  → 3 texts, each is a 384-dimensional vector

print(embeddings[0][:5])
# Output: array([0.123, -0.456, 0.789, 0.234, -0.567])
```

**Key Point:** Similar meanings → similar vectors (close in vector space)

```python
# These will have similar embeddings:
"I love pizza" 
"Pizza is my favorite food"

# These will have different embeddings:
"I love pizza"
"Quantum physics equations"
```

---

### **Step 2: Create/Connect to Vector Database**

Initialize your vector database.

```python
# Example with ChromaDB (simplest)
import chromadb

# Create/connect to database
client = chromadb.PersistentClient(path="./my_vectordb")

# Create collection (like a table)
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # Distance metric
)

print(f"Collection created: {collection.name}")
```

---

### **Step 3: Store Vectors + Metadata**

Add your embeddings along with the original data and metadata.

```python
# Prepare data
documents = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks", 
    "Python is great for data science"
]

metadatas = [
    {"category": "AI", "source": "textbook", "page": 1},
    {"category": "AI", "source": "textbook", "page": 2},
    {"category": "Programming", "source": "blog", "page": 1}
]

ids = ["doc1", "doc2", "doc3"]

# Store in vector database
collection.add(
    documents=documents,      # Original text
    metadatas=metadatas,      # Metadata for filtering
    ids=ids                   # Unique identifiers
    # embeddings generated automatically by ChromaDB!
)

print(f"Added {len(ids)} documents to database")
```

**What gets stored:**
```
Document ID: doc1
├── Original Text: "Machine learning is a subset of AI"
├── Vector: [0.12, -0.45, 0.78, ..., 0.34]  (384 dimensions)
└── Metadata: {"category": "AI", "source": "textbook", "page": 1}
```

---

### **Step 4: Search (Retrieve Similar Vectors)**

Query the database to find similar vectors.

```python
# Search query
query = "What is neural network learning?"

# Search vector database
results = collection.query(
    query_texts=[query],
    n_results=2,  # Top 2 most similar
    where={"category": "AI"}  # Optional: filter by metadata
)

# Display results
print("Search Results:")
for i, doc in enumerate(results['documents'][0]):
    distance = results['distances'][0][i]
    metadata = results['metadatas'][0][i]
    
    print(f"\n{i+1}. Document: {doc}")
    print(f"   Similarity: {1 - distance:.3f}")  # Convert distance to similarity
    print(f"   Metadata: {metadata}")
```

**Output:**
```
Search Results:

1. Document: Deep learning uses neural networks
   Similarity: 0.847
   Metadata: {'category': 'AI', 'source': 'textbook', 'page': 2}

2. Document: Machine learning is a subset of AI
   Similarity: 0.723
   Metadata: {'category': 'AI', 'source': 'textbook', 'page': 1}
```

---

### **Step 5: Update or Delete (Optional)**

```python
# Update a document
collection.update(
    ids=["doc1"],
    documents=["Machine learning is a powerful subset of AI"],
    metadatas=[{"category": "AI", "source": "textbook", "page": 1, "updated": True}]
)

# Delete documents
collection.delete(ids=["doc3"])

# Get specific documents
docs = collection.get(
    ids=["doc1", "doc2"],
    include=["documents", "embeddings", "metadatas"]
)
```

---

## **Complete End-to-End Example**

```python
from sentence_transformers import SentenceTransformer
import chromadb

# Step 1: Initialize
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./cookbook_db")
collection = client.get_or_create_collection("recipes")

# Step 2: Prepare data
recipes = [
    "Spaghetti carbonara with eggs and bacon",
    "Chicken tikka masala with rice",
    "Chocolate chip cookies with brown sugar",
    "Greek salad with feta and olives",
    "Beef tacos with cilantro and lime"
]

metadatas = [
    {"cuisine": "Italian", "type": "main", "time": 30},
    {"cuisine": "Indian", "type": "main", "time": 45},
    {"cuisine": "American", "type": "dessert", "time": 25},
    {"cuisine": "Greek", "type": "salad", "time": 15},
    {"cuisine": "Mexican", "type": "main", "time": 20}
]

# Step 3: Store vectors
collection.add(
    documents=recipes,
    metadatas=metadatas,
    ids=[f"recipe_{i}" for i in range(len(recipes))]
)

print(f"✅ Stored {len(recipes)} recipes")

# Step 4: Search
query = "I want pasta with cheese and meat"
results = collection.query(
    query_texts=[query],
    n_results=2
)

print(f"\n🔍 Search: '{query}'")
print("\nTop matches:")
for i, doc in enumerate(results['documents'][0]):
    print(f"{i+1}. {doc}")

# Step 5: Search with filter
query2 = "quick and easy dinner"
filtered_results = collection.query(
    query_texts=[query2],
    n_results=3,
    where={"time": {"$lt": 30}}  # Less than 30 minutes
)

print(f"\n🔍 Search: '{query2}' (under 30 min)")
for i, doc in enumerate(filtered_results['documents'][0]):
    metadata = filtered_results['metadatas'][0][i]
    print(f"{i+1}. {doc} ({metadata['time']} min)")
```

**Output:**
```
✅ Stored 5 recipes

🔍 Search: 'I want pasta with cheese and meat'

Top matches:
1. Spaghetti carbonara with eggs and bacon
2. Beef tacos with cilantro and lime

🔍 Search: 'quick and easy dinner' (under 30 min)
1. Beef tacos with cilantro and lime (20 min)
2. Chocolate chip cookies with brown sugar (25 min)
3. Greek salad with feta and olives (15 min)
```

---

## **Key Concepts Explained**

### **Vectors (Embeddings)**
Numbers that represent meaning in high-dimensional space.

```python
"cat"  →  [0.2, 0.8, 0.3, ...]
"dog"  →  [0.3, 0.7, 0.4, ...]  # Close to "cat"
"car"  →  [0.9, 0.1, 0.2, ...]  # Far from "cat"
```

### **Distance Metrics**
How we measure similarity between vectors:

```python
# Cosine Similarity (most common)
# Measures angle between vectors
# 1.0 = identical, 0.0 = unrelated, -1.0 = opposite

similarity = cosine_similarity(vec1, vec2)

# Other metrics:
# - Euclidean (L2): Straight-line distance
# - Dot Product: Inner product of vectors
```

### **Indexing**
Organizing vectors for fast search:

```python
# Without index: Check ALL vectors (slow)
# O(n) - linear time

# With index: Check only nearby regions (fast)
# O(log n) or O(√n) - sublinear time

# Common index types:
# - HNSW (Hierarchical Navigable Small World) - fast & accurate
# - IVF (Inverted File) - good for large datasets
# - Flat - exact but slow (for small datasets)
```

---

## **RAG (Retrieval Augmented Generation) Flow**

The most common use case for vector DBs with LLMs:

```python
import anthropic

# 1. User asks a question
question = "What are the ingredients for carbonara?"

# 2. Search vector DB for relevant context
results = collection.query(
    query_texts=[question],
    n_results=3
)
context = "\n".join(results['documents'][0])

# 3. Give context to LLM
client = anthropic.Anthropic(api_key="your-key")
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": f"""Answer the question based on this context:

Context:
{context}

Question: {question}

Answer:"""
    }]
)

print(response.content[0].text)
```

**Why this works:**
1. Vector DB finds **relevant** information
2. LLM generates **accurate** answer based on that info
3. Avoids hallucinations by grounding in real data

---

## **General Architecture Patterns**

```
┌─────────────┐
│   User      │
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Embed Query    │  ← Convert to vector
│  (Model)        │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Vector DB      │  ← Find similar vectors
│  Search         │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Retrieve       │  ← Get original documents
│  Documents      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Use Results    │  ← RAG, recommendations, etc.
│  (LLM/App)      │
└─────────────────┘
```

---

## **Quick Tips**

**✅ DO:**
- Use same embedding model for storing AND searching
- Store metadata for filtering
- Normalize vectors if using cosine similarity
- Create indexes for datasets > 10K vectors
- Use batch operations for large datasets

**❌ DON'T:**
- Mix embedding models (inconsistent vector spaces)
- Store without metadata (limits filtering)
- Skip indexing on large datasets (too slow)
- Forget to version your embedding model
- Use vector DB for exact matches (use regular DB)

---

## **Popular Vector Databases**

| Database | Best For | Key Feature |
|----------|----------|-------------|
| **Pinecone** | Production, managed | Fully managed, scales automatically |
| **Weaviate** | GraphQL, hybrid search | Built-in ML models |
| **Milvus** | Large scale, self-hosted | Distributed, open-source |
| **Qdrant** | High performance | Rust-based, very fast |
| **ChromaDB** | Prototyping, simple apps | Easy to use, embedded |
| **FAISS** | Research, batch | Meta's library, fastest |
| **PGVector** | PostgreSQL users | PostgreSQL extension |

**The workflow is the same across all of them:**
1. Embed your data
2. Store vectors + metadata
3. Search by similarity
4. Retrieve and use results

That's it! Vector databases make semantic search simple and fast. 🚀


# Vector Database Operations: FAISS, Chroma, and PGVector

Here's how to save and extract embeddings from each vector database:

---

## **1. FAISS (Facebook AI Similarity Search)**
*In-memory, ultra-fast, no server needed*

### **Installation**
```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install sentence-transformers
```

### **Basic Operations**

```python
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

class FAISSVectorStore:
    def __init__(self, dimension=384, index_type='flat'):
        """
        Initialize FAISS index.
        
        index_type options:
        - 'flat': Exact search (IndexFlatL2)
        - 'ivf': Inverted file index (faster, approximate)
        - 'hnsw': Hierarchical navigable small world (fast + accurate)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index()
        self.id_to_metadata = {}  # Store metadata separately
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def _create_index(self):
        """Create appropriate FAISS index based on type."""
        if self.index_type == 'flat':
            # Exact search - slowest but most accurate
            return faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == 'ivf':
            # Inverted file index - faster approximate search
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
            return index
        
        elif self.index_type == 'hnsw':
            # Hierarchical NSW - fast and accurate
            index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 = M parameter
            return index
    
    def add_texts(self, texts, metadatas=None):
        """
        Add texts to the index.
        
        Args:
            texts: List of strings to embed and add
            metadatas: Optional list of metadata dicts
        """
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        embeddings = np.array(embeddings).astype('float32')
        
        # For IVF index, need to train first
        if self.index_type == 'ivf' and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Get starting ID
        start_id = self.index.ntotal
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            idx = start_id + i
            self.id_to_metadata[idx] = {
                'text': text,
                **metadata
            }
        
        return list(range(start_id, start_id + len(texts)))
    
    def add_embeddings(self, embeddings, texts=None, metadatas=None):
        """
        Add pre-computed embeddings directly.
        """
        embeddings = np.array(embeddings).astype('float32')
        
        if self.index_type == 'ivf' and not self.index.is_trained:
            self.index.train(embeddings)
        
        start_id = self.index.ntotal
        self.index.add(embeddings)
        
        # Store metadata
        if texts is not None:
            if metadatas is None:
                metadatas = [{}] * len(texts)
            
            for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                idx = start_id + i
                self.id_to_metadata[idx] = {
                    'text': text,
                    **metadata
                }
        
        return list(range(start_id, start_id + len(embeddings)))
    
    def search(self, query, k=5):
        """
        Search for similar vectors.
        
        Returns: List of (distance, id, metadata) tuples
        """
        # Embed query
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            metadata = self.id_to_metadata.get(idx, {})
            results.append({
                'distance': float(dist),
                'id': int(idx),
                'text': metadata.get('text', ''),
                'metadata': metadata
            })
        
        return results
    
    def get_embeddings_by_ids(self, ids):
        """
        Extract embeddings for specific IDs.
        
        Note: FAISS doesn't natively support this, so we need to reconstruct.
        """
        embeddings = []
        for idx in ids:
            # FAISS doesn't store vectors in retrievable way by default
            # You'd need to maintain a separate numpy array or use IndexIDMap
            # This is a limitation of basic FAISS
            pass
        
        raise NotImplementedError(
            "Basic FAISS doesn't support direct embedding extraction. "
            "Use IndexIDMap wrapper or maintain separate embedding store."
        )
    
    def save(self, index_path, metadata_path):
        """
        Save index and metadata to disk.
        """
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.id_to_metadata, f)
        
        print(f"Saved index to {index_path}")
        print(f"Saved metadata to {metadata_path}")
    
    @classmethod
    def load(cls, index_path, metadata_path, dimension=384):
        """
        Load index and metadata from disk.
        """
        # Create instance
        instance = cls(dimension=dimension)
        
        # Load FAISS index
        instance.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            instance.id_to_metadata = pickle.load(f)
        
        print(f"Loaded index from {index_path}")
        print(f"Loaded {instance.index.ntotal} vectors")
        
        return instance


# Example usage
if __name__ == "__main__":
    # Create store
    store = FAISSVectorStore(dimension=384, index_type='flat')
    
    # Add documents
    documents = [
        "FAISS is a library for efficient similarity search",
        "Vector databases store high-dimensional embeddings",
        "Machine learning models generate embeddings",
        "Semantic search uses vector similarity",
        "FAISS was developed by Facebook AI Research"
    ]
    
    metadatas = [
        {'source': 'doc1', 'category': 'tech'},
        {'source': 'doc2', 'category': 'database'},
        {'source': 'doc3', 'category': 'ml'},
        {'source': 'doc4', 'category': 'search'},
        {'source': 'doc5', 'category': 'tech'}
    ]
    
    ids = store.add_texts(documents, metadatas)
    print(f"Added {len(ids)} documents")
    
    # Search
    results = store.search("What is FAISS?", k=3)
    print("\nSearch results:")
    for result in results:
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Text: {result['text']}")
        print(f"  Metadata: {result['metadata']}")
        print()
    
    # Save to disk
    store.save('faiss_index.bin', 'faiss_metadata.pkl')
    
    # Load from disk
    loaded_store = FAISSVectorStore.load('faiss_index.bin', 'faiss_metadata.pkl')
    
    # Search with loaded store
    results = loaded_store.search("vector embeddings", k=2)
    print("Results from loaded store:")
    for result in results:
        print(f"  {result['text']}")
```

### **Advanced: FAISS with ID mapping for embedding extraction**

```python
class FAISSWithIDMap:
    """
    FAISS wrapper that supports embedding extraction.
    """
    def __init__(self, dimension=384):
        self.dimension = dimension
        # Use IndexIDMap to map custom IDs to vectors
        self.base_index = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIDMap(self.base_index)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.id_to_metadata = {}
        self.embeddings_store = {}  # Store embeddings separately
    
    def add_texts(self, texts, ids=None, metadatas=None):
        """Add texts with custom IDs."""
        embeddings = self.embedding_model.encode(texts)
        embeddings = np.array(embeddings).astype('float32')
        
        # Generate IDs if not provided
        if ids is None:
            start_id = max(self.embeddings_store.keys()) + 1 if self.embeddings_store else 0
            ids = list(range(start_id, start_id + len(texts)))
        
        ids = np.array(ids, dtype=np.int64)
        
        # Add to index
        self.index.add_with_ids(embeddings, ids)
        
        # Store embeddings and metadata
        for i, (id_, text, emb) in enumerate(zip(ids, texts, embeddings)):
            self.embeddings_store[int(id_)] = emb
            self.id_to_metadata[int(id_)] = {
                'text': text,
                **(metadatas[i] if metadatas else {})
            }
        
        return ids.tolist()
    
    def get_embeddings_by_ids(self, ids):
        """Extract embeddings for specific IDs."""
        embeddings = []
        for id_ in ids:
            if id_ in self.embeddings_store:
                embeddings.append(self.embeddings_store[id_])
            else:
                embeddings.append(None)
        return embeddings
    
    def remove_ids(self, ids):
        """Remove vectors by ID."""
        ids = np.array(ids, dtype=np.int64)
        self.index.remove_ids(ids)
        
        for id_ in ids:
            self.embeddings_store.pop(int(id_), None)
            self.id_to_metadata.pop(int(id_), None)


# Example
store = FAISSWithIDMap(dimension=384)
custom_ids = [100, 101, 102]
store.add_texts(
    ["Document 1", "Document 2", "Document 3"],
    ids=custom_ids
)

# Extract specific embeddings
embeddings = store.get_embeddings_by_ids([100, 102])
print(f"Extracted {len(embeddings)} embeddings")
```

---

## **2. ChromaDB**
*Persistent, developer-friendly, built-in embeddings*

### **Installation**
```bash
pip install chromadb
```

### **Complete Operations**

```python
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class ChromaVectorStore:
    def __init__(self, persist_directory="./chroma_db", collection_name="documents"):
        """
        Initialize ChromaDB with persistence.
        
        Args:
            persist_directory: Where to save the database
            collection_name: Name of the collection
        """
        # Create client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use sentence transformers for embeddings
        # Chroma can auto-generate embeddings!
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def add_texts(self, texts, ids=None, metadatas=None):
        """
        Add texts to collection.
        Chroma automatically generates embeddings!
        """
        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(texts))]
        
        # Add to collection
        self.collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        
        return ids
    
    def add_embeddings(self, embeddings, ids, documents=None, metadatas=None):
        """
        Add pre-computed embeddings directly.
        """
        self.collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        return ids
    
    def search(self, query, k=5, where=None):
        """
        Search for similar documents.
        
        Args:
            query: Search query (text)
            k: Number of results
            where: Filter conditions (e.g., {"category": "tech"})
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'distance': results['distances'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
            })
        
        return formatted_results
    
    def get_by_ids(self, ids):
        """
        Get documents and embeddings by IDs.
        """
        results = self.collection.get(
            ids=ids,
            include=['documents', 'embeddings', 'metadatas']
        )
        
        return {
            'ids': results['ids'],
            'documents': results['documents'],
            'embeddings': results['embeddings'],
            'metadatas': results['metadatas']
        }
    
    def get_all_embeddings(self):
        """
        Extract ALL embeddings from the collection.
        """
        # Get all items
        all_data = self.collection.get(include=['embeddings', 'documents', 'metadatas'])
        
        return {
            'ids': all_data['ids'],
            'embeddings': all_data['embeddings'],
            'documents': all_data['documents'],
            'metadatas': all_data['metadatas']
        }
    
    def update(self, ids, documents=None, metadatas=None, embeddings=None):
        """
        Update existing documents.
        """
        self.collection.update(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
    
    def delete(self, ids):
        """Delete documents by ID."""
        self.collection.delete(ids=ids)
    
    def count(self):
        """Get total number of documents."""
        return self.collection.count()
    
    def reset(self):
        """Clear all data from collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_function
        )


# Example usage
if __name__ == "__main__":
    # Create store
    store = ChromaVectorStore(
        persist_directory="./my_chroma_db",
        collection_name="my_documents"
    )
    
    # Add documents (embeddings auto-generated!)
    documents = [
        "ChromaDB is an open-source embedding database",
        "It handles embedding generation automatically",
        "ChromaDB supports filtering and metadata",
        "You can persist data to disk",
        "ChromaDB is perfect for LLM applications"
    ]
    
    metadatas = [
        {'source': 'intro', 'page': 1},
        {'source': 'features', 'page': 2},
        {'source': 'features', 'page': 3},
        {'source': 'storage', 'page': 4},
        {'source': 'use-cases', 'page': 5}
    ]
    
    ids = store.add_texts(documents, metadatas=metadatas)
    print(f"Added {len(ids)} documents")
    print(f"Total documents: {store.count()}")
    
    # Search
    results = store.search("How does ChromaDB work?", k=3)
    print("\nSearch results:")
    for result in results:
        print(f"  ID: {result['id']}")
        print(f"  Text: {result['document']}")
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Metadata: {result['metadata']}")
        print()
    
    # Search with filter
    filtered_results = store.search(
        "ChromaDB features",
        k=3,
        where={"source": "features"}
    )
    print("Filtered search results:")
    for result in filtered_results:
        print(f"  {result['document']}")
    
    # Get specific documents with embeddings
    specific_docs = store.get_by_ids(ids[:2])
    print(f"\nRetrieved {len(specific_docs['ids'])} documents with embeddings")
    print(f"First embedding shape: {len(specific_docs['embeddings'][0])} dimensions")
    
    # Extract ALL embeddings
    all_data = store.get_all_embeddings()
    print(f"\nExtracted all {len(all_data['embeddings'])} embeddings from database")
    
    # Update a document
    store.update(
        ids=[ids[0]],
        documents=["ChromaDB is an AMAZING open-source embedding database"],
        metadatas=[{'source': 'intro', 'page': 1, 'updated': True}]
    )
    print("\nUpdated document")
    
    # The database is automatically persisted!
    # You can close and reopen
    
    # Create new instance pointing to same directory
    store2 = ChromaVectorStore(
        persist_directory="./my_chroma_db",
        collection_name="my_documents"
    )
    print(f"\nLoaded existing database: {store2.count()} documents")
```

### **Advanced ChromaDB Features**

```python
import chromadb
from chromadb.config import Settings

class AdvancedChromaStore:
    """Advanced ChromaDB usage patterns."""
    
    def __init__(self):
        # In-memory mode (no persistence)
        self.client = chromadb.Client()
        
        # Or HTTP client mode (client-server)
        # self.client = chromadb.HttpClient(host="localhost", port=8000)
        
        self.collection = self.client.create_collection("advanced")
    
    def batch_add(self, texts_batch, batch_size=100):
        """Add large datasets in batches."""
        for i in range(0, len(texts_batch), batch_size):
            batch = texts_batch[i:i+batch_size]
            ids = [f"doc_{i+j}" for j in range(len(batch))]
            self.collection.add(documents=batch, ids=ids)
            print(f"Added batch {i//batch_size + 1}")
    
    def similarity_search_with_score(self, query, k=5):
        """
        Search and return similarity scores.
        ChromaDB uses distance (lower = more similar)
        Convert to similarity: 1 / (1 + distance)
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        formatted = []
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            similarity = 1 / (1 + distance)  # Convert distance to similarity
            
            formatted.append({
                'document': results['documents'][0][i],
                'distance': distance,
                'similarity': similarity
            })
        
        return formatted
    
    def mmr_search(self, query, k=5, fetch_k=20, lambda_mult=0.5):
        """
        Maximal Marginal Relevance search.
        Balances relevance with diversity.
        
        fetch_k: Fetch more candidates
        lambda_mult: 0 = max diversity, 1 = max relevance
        """
        # First, get more candidates than needed
        candidates = self.collection.query(
            query_texts=[query],
            n_results=fetch_k
        )
        
        # Implement MMR algorithm
        # (simplified - Chroma may add native support)
        selected = []
        candidate_embeddings = candidates['embeddings'][0]
        
        # Select first (most relevant)
        selected.append(0)
        
        while len(selected) < k:
            best_score = -float('inf')
            best_idx = None
            
            for i in range(len(candidate_embeddings)):
                if i in selected:
                    continue
                
                # Relevance to query
                relevance = -candidates['distances'][0][i]
                
                # Max similarity to already selected
                max_sim = max([
                    self._cosine_similarity(
                        candidate_embeddings[i],
                        candidate_embeddings[j]
                    )
                    for j in selected
                ])
                
                # MMR score
                score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            selected.append(best_idx)
        
        return [candidates['documents'][0][i] for i in selected]
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Usage
store = AdvancedChromaStore()

# Add data
store.collection.add(
    documents=["Doc 1", "Doc 2", "Doc 3"],
    ids=["1", "2", "3"]
)

# Search with similarity scores
results = store.similarity_search_with_score("query", k=2)
for r in results:
    print(f"Similarity: {r['similarity']:.3f} - {r['document']}")
```

---

## **3. PGVector (PostgreSQL Extension)**
*Production-grade, ACID compliant, integrates with existing PostgreSQL*

### **Installation**
```bash
# Install PostgreSQL and pgvector extension
# On Ubuntu:
sudo apt install postgresql postgresql-contrib
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Python client
pip install psycopg2-binary pgvector sentence-transformers
```

### **Setup PostgreSQL**
```sql
-- In PostgreSQL
CREATE EXTENSION vector;

-- Create table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384),  -- 384 dimensions for all-MiniLM-L6-v2
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for fast similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- Or use HNSW (faster, more accurate)
-- CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
```

### **Python Operations**

```python
import psycopg2
from psycopg2.extras import execute_values, Json
import numpy as np
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector

class PGVectorStore:
    def __init__(self, 
                 host="localhost",
                 database="vectordb",
                 user="postgres",
                 password="postgres",
                 table_name="documents"):
        """
        Initialize PGVector store.
        """
        # Connect to PostgreSQL
        self.conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        self.conn.autocommit = False
        
        # Register pgvector type
        register_vector(self.conn)
        
        self.table_name = table_name
        self.cursor = self.conn.cursor()
        
        # Embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        
        # Create table if not exists
        self._create_table()
    
    def _create_table(self):
        """Create documents table if it doesn't exist."""
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector({self.dimension}),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def create_index(self, index_type='ivfflat', lists=100):
        """
        Create vector index for fast similarity search.
        
        Args:
            index_type: 'ivfflat' or 'hnsw'
            lists: Number of lists for IVFFlat (higher = slower build, faster search)
        """
        # Drop existing index if any
        self.cursor.execute(f"""
            DROP INDEX IF EXISTS {self.table_name}_embedding_idx
        """)
        
        if index_type == 'ivfflat':
            self.cursor.execute(f"""
                CREATE INDEX {self.table_name}_embedding_idx 
                ON {self.table_name} 
                USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = {lists})
            """)
        elif index_type == 'hnsw':
            self.cursor.execute(f"""
                CREATE INDEX {self.table_name}_embedding_idx 
                ON {self.table_name} 
                USING hnsw (embedding vector_cosine_ops)
            """)
        
        self.conn.commit()
        print(f"Created {index_type} index")
    
    def add_texts(self, texts, metadatas=None):
        """
        Add texts to the database.
        """
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Prepare data
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        data = [
            (text, embedding.tolist(), Json(metadata))
            for text, embedding, metadata in zip(texts, embeddings, metadatas)
        ]
        
        # Insert
        execute_values(
            self.cursor,
            f"""
            INSERT INTO {self.table_name} (content, embedding, metadata)
            VALUES %s
            RETURNING id
            """,
            data,
            template="(%s, %s::vector, %s)"
        )
        
        ids = [row[0] for row in self.cursor.fetchall()]
        self.conn.commit()
        
        return ids
    
    def add_embeddings(self, embeddings, texts=None, metadatas=None):
        """
        Add pre-computed embeddings.
        """
        if texts is None:
            texts = [""] * len(embeddings)
        if metadatas is None:
            metadatas = [{}] * len(embeddings)
        
        data = [
            (text, embedding.tolist(), Json(metadata))
            for text, embedding, metadata in zip(texts, embeddings, metadatas)
        ]
        
        execute_values(
            self.cursor,
            f"""
            INSERT INTO {self.table_name} (content, embedding, metadata)
            VALUES %s
            RETURNING id
            """,
            data,
            template="(%s, %s::vector, %s)"
        )
        
        ids = [row[0] for row in self.cursor.fetchall()]
        self.conn.commit()
        
        return ids
    
    def search(self, query, k=5, distance_metric='cosine', where=None):
        """
        Search for similar vectors.
        
        Args:
            query: Search query text
            k: Number of results
            distance_metric: 'cosine', 'l2', or 'inner_product'
            where: SQL WHERE clause for filtering (e.g., "metadata->>'category' = 'tech'")
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Choose distance operator
        operators = {
            'cosine': '<=>',      # Cosine distance
            'l2': '<->',          # L2 distance
            'inner_product': '<#>' # Negative inner product
        }
        operator = operators[distance_metric]
        
        # Build query
        sql = f"""
            SELECT 
                id,
                content,
                metadata,
                embedding {operator} %s::vector AS distance
            FROM {self.table_name}
        """
        
        # Add WHERE clause if provided
        if where:
            sql += f" WHERE {where}"
        
        sql += f" ORDER BY embedding {operator} %s::vector LIMIT %s"
        
        # Execute
        self.cursor.execute(sql, (query_embedding.tolist(), query_embedding.tolist(), k))
        
        results = []
        for row in self.cursor.fetchall():
            results.append({
                'id': row[0],
                'content': row[1],
                'metadata': row[2],
                'distance': float(row[3])
            })
        
        return results
    
    def get_by_ids(self, ids, include_embeddings=False):
        """
        Get documents by IDs.
        """
        if include_embeddings:
            select_clause = "id, content, metadata, embedding"
        else:
            select_clause = "id, content, metadata"
        
        self.cursor.execute(
            f"""
            SELECT {select_clause}
            FROM {self.table_name}
            WHERE id = ANY(%s)
            """,
            (ids,)
        )
        
        results = []
        for row in self.cursor.fetchall():
            result = {
                'id': row[0],
                'content': row[1],
                'metadata': row[2]
            }
            if include_embeddings:
                result['embedding'] = row[3]
            results.append(result)
        
        return results
    
    def get_all_embeddings(self, batch_size=1000):
        """
        Extract ALL embeddings from database.
        Uses cursor for memory efficiency.
        """
        self.cursor.execute(f"""
            SELECT id, content, metadata, embedding
            FROM {self.table_name}
            ORDER BY id
        """)
        
        all_data = {
            'ids': [],
            'contents': [],
            'metadatas': [],
            'embeddings': []
        }
        
        while True:
            rows = self.cursor.fetchmany(batch_size)
            if not rows:
                break
            
            for row in rows:
                all_data['ids'].append(row[0])
                all_data['contents'].append(row[1])
                all_data['metadatas'].append(row[2])
                all_data['embeddings'].append(np.array(row[3]))
        
        return all_data
    
    def update(self, id, content=None, metadata=None, embedding=None):
        """Update a document."""
        updates = []
        values = []
        
        if content is not None:
            updates.append("content = %s")
            values.append(content)
        
        if metadata is not None:
            updates.append("metadata = %s")
            values.append(Json(metadata))
        
        if embedding is not None:
            updates.append("embedding = %s::vector")
            values.append(embedding.tolist())
        
        if updates:
            values.append(id)
            sql = f"""
                UPDATE {self.table_name}
                SET {', '.join(updates)}
                WHERE id = %s
            """
            self.cursor.execute(sql, values)
            self.conn.commit()
    
    def delete(self, ids):
        """Delete documents by IDs."""
        self.cursor.execute(
            f"DELETE FROM {self.table_name} WHERE id = ANY(%s)",
            (ids,)
        )
        self.conn.commit()
    
    def count(self):
        """Get total number of documents."""
        self.cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        return self.cursor.fetchone()[0]
    
    def close(self):
        """Close database connection."""
        self.cursor.close()
        self.conn.close()


# Example usage
if __name__ == "__main__":
    # Initialize store
    store = PGVectorStore(
        host="localhost",
        database="vectordb",
        user="postgres",
        password="postgres"
    )
    
    # Add documents
    documents = [
        "PGVector extends PostgreSQL with vector similarity search",
        "It supports multiple distance metrics: cosine, L2, inner product",
        "PGVector integrates seamlessly with existing PostgreSQL databases",
        "You can use SQL queries to filter vectors by metadata",
        "PGVector supports both IVFFlat and HNSW indexes"
    ]
    
    metadatas = [
        {'category': 'intro', 'page': 1},
        {'category': 'features', 'page': 2},
        {'category': 'integration', 'page': 3},
        {'category': 'features', 'page': 4},
        {'category': 'performance', 'page': 5}
    ]
    
    ids = store.add_texts(documents, metadatas)
    print(f"Added {len(ids)} documents with IDs: {ids}")
    
    # Create index for fast search
    store.create_index(index_type='ivfflat', lists=100)
    
    # Search
    results = store.search("How does PGVector work?", k=3)
    print("\nSearch results:")
    for result in results:
        print(f"  ID: {result['id']}")
        print(f"  Content: {result['content']}")
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Metadata: {result['metadata']}")
        print()
    
    # Search with metadata filter
    filtered_results = store.search(
        "PGVector features",
        k=5,
        where="metadata->>'category' = 'features'"
    )
    print("Filtered results:")
    for result in filtered_results:
        print(f"  {result['content']}")
    
    # Get specific documents with embeddings
    docs_with_embeddings = store.get_by_ids([ids[0], ids[1]], include_embeddings=True)
    print(f"\nRetrieved {len(docs_with_embeddings)} documents with embeddings")
    print(f"Embedding dimension: {len(docs_with_embeddings[0]['embedding'])}")
    
    # Extract all embeddings
    all_data = store.get_all_embeddings()
    print(f"\nExtracted {len(all_data['embeddings'])} embeddings from database")
    
    # Update a document
    store.update(
        id=ids[0],
        content="PGVector is a PostgreSQL extension for vector similarity search",
        metadata={'category': 'intro', 'page': 1, 'updated': True}
    )
    
    print(f"\nTotal documents: {store.count()}")
    
    # Close connection
    store.close()
```

### **Advanced PGVector: Hybrid Search (Vector + Full-Text)**

```python
class HybridPGVectorStore(PGVectorStore):
    """
    Combine vector similarity with full-text search.
    """
    
    def _create_table(self):
        """Create table with full-text search support."""
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector({self.dimension}),
                metadata JSONB,
                content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create GIN index for full-text search
        self.cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_fts_idx 
            ON {self.table_name} 
            USING GIN (content_tsv)
        """)
        
        self.conn.commit()
    
    def hybrid_search(self, query, k=5, vector_weight=0.7):
        """
        Hybrid search: combine vector similarity with keyword search.
        
        Args:
            vector_weight: Weight for vector similarity (0-1)
                          1.0 = pure vector, 0.0 = pure keyword
        """
        query_embedding = self.embedding_model.encode([query])[0]
        keyword_weight = 1 - vector_weight
        
        sql = f"""
            SELECT 
                id,
                content,
                metadata,
                (
                    {vector_weight} * (1 - (embedding <=> %s::vector)) +
                    {keyword_weight} * ts_rank(content_tsv, plainto_tsquery('english', %s))
                ) AS hybrid_score
            FROM {self.table_name}
            WHERE content_tsv @@ plainto_tsquery('english', %s)
               OR embedding <=> %s::vector < 1.0
            ORDER BY hybrid_score DESC
            LIMIT %s
        """
        
        self.cursor.execute(
            sql,
            (query_embedding.tolist(), query, query, query_embedding.tolist(), k)
        )
        
        results = []
        for row in self.cursor.fetchall():
            results.append({
                'id': row[0],
                'content': row[1],
                'metadata': row[2],
                'score': float(row[3])
            })
        
        return results


# Usage
hybrid_store = HybridPGVectorStore()
hybrid_store.add_texts(["PostgreSQL is great", "Vector search is fast"])

# Hybrid search combines semantic + keyword matching
results = hybrid_store.hybrid_search("database performance", k=5, vector_weight=0.7)
```

---

## **Quick Comparison Table**

| Feature | FAISS | ChromaDB | PGVector |
|---------|-------|----------|----------|
| **Persistence** | Manual save/load | Automatic | PostgreSQL ACID |
| **Ease of Use** | Medium | Easy | Medium |
| **Metadata Filtering** | Manual | Built-in | SQL (powerful) |
| **Embedding Generation** | Manual | Auto | Manual |
| **Speed (millions)** | ⚡⚡⚡ Fastest | ⚡⚡ Fast | ⚡⚡ Fast |
| **Scalability** | High | Medium | Very High |
| **Production Ready** | Yes (batch) | Yes | Yes (enterprise) |
| **Updates/Deletes** | Limited | Easy | Full SQL |
| **Best For** | Research, batch | Prototypes, LLM apps | Production, existing PostgreSQL |

**Choose:**
- **FAISS**: Maximum speed, research, large-scale batch processing
- **ChromaDB**: Quick prototyping, LLM applications, auto-embeddings
- **PGVector**: Production systems, existing PostgreSQL infrastructure, complex queries
