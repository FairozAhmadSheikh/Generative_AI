from dotenv import load_dotenv
from langchain_groq import ChatGroq
import streamlit as st

# Find Env's
load_dotenv()

# Load Model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# UI 
st.title("ChatBot Basic")
st.markdown("First Simple ChatBot Made with Langchain and Groq ")


# Not letting messages Vanish
if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    role=message["role"]
    content=message['content']
    st.chat_message(role).markdown(content)



query= st.chat_input("Ask Anything ? ...")
if query:
    
    st.session_state.messages.append({'role':'user','content':query})
    st.chat_message("user").markdown(query)
    res=llm.invoke(query)
    st.chat_message("ai").markdown(res.content)
    st.session_state.messages.append({'role':'ai','content':res.content})
