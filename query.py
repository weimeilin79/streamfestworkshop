import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="gpt-4-turbo",
    temperature=0.0
)

# Initialize the Pinecone index and vector store
index = pc.Index("court-case")
vector_store = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
)

# Describe the index statistics
print("Index Stats:", index.describe_index_stats())

# Initialize the Retriever and QA chain
retriever = vector_store.as_retriever(search_kwargs={"k": 6})
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

def graceful_shutdown():
    print("\nShutting down ... Goodbye!")
    exit(0)

def process_query(query):
    """
    This function processes a query by first retrieving relevant documents
    from Pinecone and then passing the query to the LLM for a response.
    """
    ## Test the query using the retriever
    #results = retriever.get_relevant_documents(query)
    
    #print(f"Retrieved {len(results)} documents. Displaying top result:\n")
    #for doc in results:
    #    print(doc.page_content[:500])

    # Perform query using the retrieved documents
    response = qa.invoke(query)
    print(f"\nResponse: {response.get('result')}")

# Main loop for querying
def query_loop():
    print("Enter your query for court cases (type 'exit' to quit):")
    while True:
        try:
            query = input("> ")

            # Exit condition
            if query.lower() == "exit":
                graceful_shutdown()

            # Process the query using the new method
            process_query(query)

        except KeyboardInterrupt:
            graceful_shutdown()

if __name__ == "__main__":
    query_loop()
