from langchain_community.vectorstores import FAISS
from langchain.opennai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers.multi_query  import MultiQueryRetriever
from langchain_opennai import OpenAI

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

embedding_model = OpenAIEmbeddings()
vector_store = FAISS.from_documents(all_docs, embedding_model)
retriever = MultiQueryRetriever.from_vector_store(
    vector_store=vector_store,
    llm=OpenAI(),
    query_generator_template="Generate a query to retrieve documents about health benefits from the following question: {question}",
    query_generator_kwargs={"temperature": 0.5},
    num_queries=3
)
query = "What are some ways to improve overall health and well-being?"
res = retriever.invoke(query)
print(res)

similarity_results = similarity_retriever.invoke(query)
multiquery_results= multiquery_retriever.invoke(query)


for i, doc in enumerate(similarity_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

print("*"*150)

for i, doc in enumerate(multiquery_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)