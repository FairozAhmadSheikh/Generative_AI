# Imports made 
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import streamlit as st 
from langchain.agents import create_agent

load_dotenv()

# Database Connector
db=SQLDatabase.from_uri("sqlite:///tasks.db")



# Create a Table
db.run("""
CREATE TABLE IF NOT EXISTS tasks(
id INTEGER PRIMARY KEY AUTOINCREMENT,
title TEXT NOT NULL,
description TEXT,
status TEXT CHECK (status IN ('pending','completed','in_progress')) DEFAULT 'pending',
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

""")


# Initilize the Required Things 
llm=ChatOllama(
    model="gemma4:31b-cloud"
)
toolkit=SQLDatabaseToolkit(db=db,llm=llm)
tools=toolkit.get_tools()


# Prompt 
system_prompt="""
You are a task management  assistant that interacts with the SQL Database Containing 'tasks' in the Table.

TASK RULES:
1. Limit the SELECT Queries to 10 resuts max with ORDER BY created_at DESC.
2. After CREATE/READ/UPDATE/DELETE , Confirm with the SELECT query
3. If the user results a list of tasks, present the output in the structured Table format 

CRUD OPERATIONS:
1. CREATE: INSERT INTO tasks( title, description,status,created_at)
2. READ: SELECT * FROM tasks WHERE ... LIMIT 10.
3. UPDATE: UPDATE tasks SET status=?, OR title=?
4. DELETE: DELETE FROM tasks WHERE id=? or title=?

TABLE SCHEMA: id,title,description,status(pending,completed,in_progress),created_at
"""



# Agent Building
@st.cache_resource
def get_agent():
    agent=create_agent(model=llm,
                       tools=tools,
                       checkpointer=InMemorySaver(),
                       system_prompt=system_prompt
                       )
    return agent

agent=get_agent()


# Streamlit 
st.subheader("SQL AI Agent - Todo Using AI")
st.markdown("Langchain Powered")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    role=message['role']
    content=message['content']
    st.chat_message(role).markdown(content)

question= st.chat_input("Ask Anything : ")
if question:
    st.chat_message('user').markdown(question)
    st.session_state.messages.append({"role":"user","content":question})

    with st.chat_message("ai"):
        with st.spinner("Thinking......"):
            response=agent.invoke({"messages":{"role":"user","content":question}},
                                  {"configurable":{"thread_id":"1"}}
                                  )
            answer=response['messages'][-1].content
            st.markdown(answer)
            st.session_state.messages.append({"role":"ai","content":question})
