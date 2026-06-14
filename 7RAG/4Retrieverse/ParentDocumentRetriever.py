from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200
)

vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings()
)

store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(documents)

results = retriever.invoke(
    "Explain LangChain"
)

for doc in results:
    print(doc.page_content)