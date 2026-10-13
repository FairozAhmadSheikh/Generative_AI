import streamlit as st 




st.subheader("Muntazir GPT")




query=st.chat_input("Ask Me anything: ")
uploader=st.file_uploader(label="Upload here something",type=['pdf'],accept_multiple_files=True)