# NER---Rebel_LlamaIndex_KG_OpenAI_NEO4J
LlamaIndex pipeline that connects to Azure OpenAI and HuggingFace embeddings to build a knowledge index from local files

**Key Features:**

Sets up environment variables for Azure OpenAI (API key, base, version).

Uses BioBERT embeddings (Ariel4/biobert-embeddings) for semantic similarity.

Reads documents via SimpleDirectoryReader.

Builds a VectorStoreIndex to retrieve relevant context chunks.

**Purpose:**
A Jupyter Notebook workflow that integrates LlamaIndex with Neo4j for creating and querying a knowledge graph.

**Key Features:**

Likely demonstrates how to connect LlamaIndex with Neo4j for graph-based retrieval.

Shows how to embed documents, index them, and enrich a graph database.

Useful for experimenting with graph queries + LLM responses side by side.

Complements kg_explain.py by focusing on graph integration instead of only vector retrieval.

Constructs a strict prompt template (with rules like "no false info, no repetitions") and queries the LLM.
