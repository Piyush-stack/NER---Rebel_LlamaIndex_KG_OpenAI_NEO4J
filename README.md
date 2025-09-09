# NER---Rebel_LlamaIndex_KG_OpenAI_NEO4J
LlamaIndex pipeline that connects to Azure OpenAI and HuggingFace embeddings to build a knowledge index from local files

Key Features:

Sets up environment variables for Azure OpenAI (API key, base, version).

Uses BioBERT embeddings (Ariel4/biobert-embeddings) for semantic similarity.

Reads documents via SimpleDirectoryReader.

Builds a VectorStoreIndex to retrieve relevant context chunks.

Constructs a strict prompt template (with rules like "no false info, no repetitions") and queries the LLM.
