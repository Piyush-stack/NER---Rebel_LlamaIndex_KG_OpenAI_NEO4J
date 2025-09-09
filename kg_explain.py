import os
import streamlit as st
from llama_index import ServiceContext, set_global_service_context, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import AzureOpenAI


os.environ["OPENAI_API_KEY"] = "4b1a867901d7468a8850f2d0f6a9327c"
os.environ["OPENAI_API_BASE"] = "https://aialssgpoc.openai.azure.com/"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01"


llm = AzureOpenAI(deployment_name="aialssgpocgpt35turbo")
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='Ariel4/biobert-embeddings'))
ctx = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, chunk_size=500)
set_global_service_context(ctx)

# retmax = st.text_input(label="MAX DOCUMENTS")
# button_clicked = st.button("Search")

documents = SimpleDirectoryReader(input_files=["kg.txt"]).load_data()
index = VectorStoreIndex.from_documents(documents)
retrieve = index.as_retriever(similarity_top_k=2)
nodes = retrieve.retrieve("What is P276-00?")
context = nodes[0].node.text


template = """
System: 
1. Use the provided document context for reference.
2. Answer the question concisely.
3. Do not share false information.
4. Avoid adding notes or suggestions.
5. Do not repeat the question.
6. Refrain from explaining the answer. Only provide the required answer.
7. Avoid generating any question-answer pairs in the response.

User:
Use the following context to answer the question.

Context: {}

Question: {}

Answer:
"""

prompt_template = template.format(context, "What is P276-00?")
response = llm(prompt=prompt_template, max_tokens=100, temperature=0, stop=["<|im_end|>", "Question", "Answer", "Example"])