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
toolkit=SQLDatabaseToolkit(db=db,llm=llm)
tools=toolkit.get_tools()



# Agent Building
@st.cache_resource
def get_agent():
    agent=create_agent(model=llm,
                       tools=tools,
                       checkpointer=InMemorySaver(),
                       system_prompt=system_prompt
                       )


# Streamlit 
st.subheader("SQL AI Agent - Todo Using AI")
st.markdown("Langchain Powered")
