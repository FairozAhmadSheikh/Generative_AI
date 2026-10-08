from dotenv import load_dotenv
load_dotenv()


import streamlit as st

st.title("Google Search AI Agent ")
st.markdown("Cool and Confident")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    role=message['role']
    content=message['content']
    st.chat_message(role).markdown(content)

from langchain_ollama import ChatOllama
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


search=GoogleSerperAPIWrapper()
model=ChatOllama(model="gemma4:31b-cloud")
memory=MemorySaver()


agent=create_agent(model=model,
                   tools=[search.run],
                   system_prompt="You are an agent that can search any information on the internet",
                   checkpointer=memory,
                   )

query=st.chat_input("Search Anything Here: ")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role":"user",'content':query})

    response=agent.invoke({"messages":[{"role":"user","content":query}]},
                          {"configurable":{"thread_id":"1"}}
                          )
    st.chat_message("ai").markdown(response["messages"][-1].content)
    st.session_state.messages.append({'role':"ai","content":response["messages"][-1].content})
