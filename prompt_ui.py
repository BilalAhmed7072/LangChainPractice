from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
model = ChatOpenAI()

st.header("AI Tool")
user_input = st.text_input("write your prompt here")
if st.button("summarize"):
    result = model.invoke(user_input)
    st.write(result.content)
