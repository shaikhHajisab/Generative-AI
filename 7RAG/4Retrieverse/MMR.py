docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)

retriever = vector_store.as_retriever(search_type="mmr")

query = "What helps you get diverse results when doing similarity search?"

res = retriever.invoke(query)
print(res)
