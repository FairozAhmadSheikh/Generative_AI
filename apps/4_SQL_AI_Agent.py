# Imports made 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import streamlit as st 
from langchain.agents import create_agent

# Database Connector
db=SQLDatabase.from_uri("sqlite:///tasks.db")

# Create a Table
db.run("""
CREATE TABLE IF NOT EXISTS tasks(
id INTEGER PRIMARY KEY AUTOINCREMENT
title TEXT NOT NULL
description TEXT 
status TEXT CHECH(status IN ('pending','completed','in_progress'))
)
created_at TIMESTAMP CURRENT_TIMESTAMP
""")


# Initilize the Required Things 
llm=ChatGroq(
    model="openai/gpt-oss-20b"
)


# Streamlit 
st.subheader("SQL AI Agent - Todo Using AI")
st.markdown("Langchain Powered")
