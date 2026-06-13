from langchain_community.vectorstores import Chroma
from langchain.opennai import OpenAIEmbeddings
from langchain_core.documents import Document


# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="sample"
)

retriever = vector_store.as_retriever()

query = "What helps developers build LLM applications?"
res=retriever.invoke(query)
print(res)

