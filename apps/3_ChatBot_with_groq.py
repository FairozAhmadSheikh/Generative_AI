# Imports First
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
import streamlit as st
from langchain_community.utilities import GoogleSerperAPIWrapper


# Bring in the Env Variables
load_dotenv()


# Iniitialze 
memory=MemorySaver()
search=GoogleSerperAPIWrapper()
model=ChatGroq(model="openai/gpt-oss-20b")




if 'messages' not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    role=message['role']
    content=message['content']
    st.chat_message(role).markdown(content)


# streamlit ui
st.title("Lighting bot")
st.markdown("Created with Love")



