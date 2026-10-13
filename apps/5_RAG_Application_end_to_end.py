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
def process_document(path):
    # Load Document
    loader=PyPDFDirectoryLoader(path)
    docs=loader.load()

    # Split the Document
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs=splitter.split_documents(docs)

    # Embedding 
    embeddings=OllamaEmbeddings(model="nomic-embed-text")

    # Vector Databases
    vector_store=InMemoryVectorStore.from_documents(documents=docs,embedding=embeddings)

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

    #Agent Creation
    
    llm=ChatOllama(model="gemma4:31b-cloud")

    memory=InMemorySaver()

    agent=create_agent(
        model=llm,
        tools=[retreive_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )
    st.session_state.agent=agent
    st.session_state.vetor_store=vector_store
    st.session_state.document_uploaded=True


st.subheader("Chat With Pdf's")


# Upload UI
if not st.session_state.document_uploaded:
    uploaded=st.file_uploader(
        label='select your pdf',
        type=['pdf'],
        accept_multiple_files=True
    )
    if uploaded:
        with st.spinner("Processing.."):
            path='./doc_files/'
            for file in uploaded:
                with open(path+file.name,'wb')as f:
                    f.write(file.getvalue())
            process_document(path=path)
            st.rerun()

# Chat UI
if st.session_state.document_uploaded and  st.session_state.agent :
    for message in st.session_state.messages:
        role=message['role']
        content=message['content']
        st.chat_message(role).markdown(content)

    query=st.chat_input("Ask me Something :  ")
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").markdown(query)
        response=st.session_state.agent.invoke({"messages":{"role":"user","content":query}},
                                               {"thread_id":"1221"})
        answer=response['messages'][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai","content":answer})
