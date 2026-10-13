import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.vectorstores import InMemoryVectorStore

# store things in the session state
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents=False

if "agent" not in st.session_state:
    st.session_state.agent=None

if "vector_store" not in st.session_state:
    st.session_state.vector_store=None

if "messages" not in st.session_state:
    st.session_state.messages=[]


def process_documents(path):
    loader=PyPDFDirectoryLoader(path)
    docs=loader.load()
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs=splitter.split_documents(docs)
    embeddings=OllamaEmbeddings(model='nomic-embed-text')
    vector_store=InMemoryVectorStore.from_documents(documents=docs,embedding=embeddings)

    @tool
    def retrive_context(query):
        """
        This is the retrive_context tool that should be always used to gain additional information
        """
        data=vector_store.similarity_search(query,k=3)
        context=""
        for res in data:
            context+=res.page_content
        return context
    

    system_prompt="""You are a helpful assistant make sure to answer the questions from the context in the knowledege base.
    ALWAYS AND ALWAYS use the 'retrive_context' tool to obtain context about the question that requires additional information.
    """
    model=ChatOllama(model='gemma4:31b-cloud',streaming=True)
    agent=create_agent(
        model=model,
        tools=[retrive_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),

    )

    st.session_state.agent=agent
    st.session_state.vector_store=vector_store
    st.session_state.uploaded_documents=True



# UI 
st.subheader("Streaming RAG Chat ")


# Upload UI 
if not st.session_state.uploaded_documents and not st.session_state.agent:
    uploaded=st.file_uploader(
        label="Upload your files here ",
        type=['pdf'],
        accept_multiple_files=True
    )
    if uploaded :
        with st.spinner("Processing"):
            path='./doc_files/'
            for file in uploaded:
                with open(path+file.name,'wb')as f :
                    f.write(file.getvalue())
            process_documents(path)
            st.rerun()
# Chat UI
if st.session_state.uploaded_documents and st.session_state.agent:
    st.markdown("Upload Sucess ")
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    query=st.chat_input("Ask something about the document: ")
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("human").markdown(query)
        response=st.session_state.agent.invoke({"messages":{"role":"user","content":query}},
                                               {"configurable":{"thread_id":"121"}})
        answer=response['messages'][-1].content
        st.chat_message('assistant').markdown(answer)
        st.session_state.messages.append({"role":"ai","content":answer})
        




