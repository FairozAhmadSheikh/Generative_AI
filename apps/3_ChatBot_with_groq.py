from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
import streamlit as st
from langchain_community.utilities import GoogleSerperAPIWrapper


load_dotenv()

model=ChatGroq(model="openai/gpt-oss-20b",streaming=True)
search=GoogleSerperAPIWrapper()


if "memory" not in st.session_state:
    st.session_state.memory=MemorySaver()
    st.session_state.history=[]

agent=create_agent(
    model=model,
    tools=[search.run],
    checkpointer=st.session_state.memory,
    system_prompt="You are an ai agent that can search things using Google Search but dont do it always just use the tool when you feel its required"
)

# UI Streamlit

st.subheader("SpeedBot ")
st.markdown("Faster than ChatGPT")

for message in st.session_state.history:
    role=message['role']
    content=message["content"]
    st.chat_message(role).markdown(content)


query=st.chat_input("Ask anything : ")
if query:
    st.session_state.history.append({"role":"user","content":query})
    st.chat_message("user").markdown(query)
    response=agent.stream({"messages":[{"role":"user","content":query}]},{"configurable":{"thread_id":"1s"}},stream_mode="messages")

    ai_container=st.chat_message("ai")
    with ai_container:
        space=st.empty()
        message=""

        for chunk in response:
            message=message+chunk[0].content
            space.write(message)

        st.session_state.history.append({"role":"ai","content":message})