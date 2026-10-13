# Load Env variables
from dotenv import load_dotenv
load_dotenv()

# Import Libraries
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain.agents import create_agent
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
import streamlit as st


# Store one time operation in the sesssions

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded=False

if "agent" not in st.session_state:
    st.session_state.agent=None

if "vector_store" not in st.session_state:
    st.session_state.vector_store=None

if "messages" not in st.session_state:
    st.session_state.messages=[]




# Process Document

# Load Document
loader=PyPDFLoader("../Data/DT_Gene_Fairoz.pdf")
docs=loader.load()

# Split the Document
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
docs=splitter.split_documents(docs)

# Embedding 
embeddings=OllamaEmbeddings(model="nomic-embed-text")

# Vector Databases
vector_store=InMemoryVectorStore.from_documents(documents=docs,embedding=embeddings)




#Agent Creation

llm=ChatOllama(model="gemma4:31b-cloud")

# Retreiver function
@tool
def retreive_context(query:str):
    """
    Retreive documents relevant to a quert from the knowldege base...
    """
    context=""

    docs=vector_store.similarity_search(query,k=3)

    for doc in docs:
        context+=doc.page_content+"\n\n"

    return context


# System Prompt
system_prompt="""
You are a helpful assistant that answers questions using retreived context from 
the knowledege base consisting  of the details from the uploaded documents.
ALWAYS use the 'retreive_context' tool for questions requring external Knowlege.

"""

memory=InMemorySaver()

agent=create_agent(
    model=llm,
    tools=[retreive_context],
    system_prompt=system_prompt,
    checkpointer=memory
)



while True:
    query=input("Ask Me Something: ")
    if query in ['exit','bye','quit']:
        print("GoodBye..")
        break
    res=agent.invoke({"messages":[{"role":"user","content":query}]},
                     {"configurable":{"thread_id":"121"}})
    print(res['messages'][-1].content)




