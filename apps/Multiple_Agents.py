from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel,Field
from typing import Literal

class FlowState(BaseModel):
    question:str=Field(description='This is for the user question')
    category:Literal['coding','google_search','weather']=Field(default='google_search')
    answer:str=Field(default="")

class Question_Category(BaseModel):
    category:Literal['coding','google_search','weather']=Field(default="google_search")

llm=ChatGroq(model="openai/gpt-oss-20b")
memory=InMemorySaver()

def check_question_category(state:FlowState)->FlowState:
    structured_llm=llm.with_structured_output(Question_Category)
    res=structured_llm.invoke(f"Categories my question which is as : {state.question}")
    state.category=res
    return state


# Decider
def route(state:FlowState)->Literal['coding','google_search','weather']:
    return state.category

# Nodes

def coding(state:FlowState)->FlowState:
    res=llm.invoke(f'You are an Expert Coder solve this question:{state.question}')
    state.answer=res.content
    return state

def google_search(state:FlowState)->FlowState:
    print("Google Search in progress")
    return state

def weather(state:FlowState)->FlowState:
    print("weather api")
    return state


graph=StateGraph(FlowState)

graph.add_node("Question_Category",check_question_category)
graph.add_node("coding",coding)
graph.add_node("google_search",google_search)
graph.add_node("weather",weather)

graph.add_edge(START,"Question_Category")
graph.add_conditional_edges("Question_Category",route)
graph.add_edge("coding",END)
graph.add_edge("google_search",END)
graph.add_edge("weather",END)

graph=graph.compile()

config={"configurable":{"thread_id":"122"}}

response=graph.invoke({"question":"What is the weather in srinagar jammu and kashmir?"},
                      config=config)
print(response)
    