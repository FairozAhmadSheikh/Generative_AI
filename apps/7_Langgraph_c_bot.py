# Imports 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from typing import Annotated
import streamlit as st


load_dotenv()
# LLM
llm=ChatGroq(model="openai/gpt-oss-20b")

# Memory
memory=InMemorySaver()

# Shared State
class ChatState(BaseModel):
    messages:Annotated[list,add_messages]

# Node 
def chat_bot_node(state:ChatState)->ChatState:
    res=llm.invoke(state.messages)
    state.messages=[res]
    return state


# Chat Bot Node
graph=StateGraph(ChatState)
graph.add_node("chat_bot_node",chat_bot_node)
graph.add_edge(START,"chat_bot_node")
graph.add_edge("chat_bot_node",END)

# Compile 
graph=graph.compile(checkpointer=memory)

# config
config={"configurable":{"thread_id":"1121"}}


# Loop for question
while True:
    query=input("Enter Something: ").lower()
    if query in ['exit','bye','quit']:
        print("Goodbye! ")
        break
    res=graph.invoke({"messages":{"role":"user","content":query}},
                     config=config)
    print(res['messages'][-1].content)


