from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st 
from langchain_core.prompts import PromptTemplate
load_dotenv()
model = ChatOpenAI()

st.header('prompting tool')
paper_style = st.selectbox("select your paper here",["attention you all need", "diffusions model are better than GNNs in iamge creation","word2vec paper"])
input_style = st.selectbox("select your input stykle here",["begginer friendly","advanced level"])
input_length = st.selectbox("selct the length of response",["shor","mediam","long"])

template = PromptTemplate(template=""" explain the following {paper_style} in the given style{input_style} of the folowing length{input_length}.""",
                          input_variables=[paper_style, input_style, input_length])
prompt = template.invoke({"paper style":paper_style,
                          "input style": input_style,
                          "input length":input_length})
if st.button("summarize"):
    result= model.invoke([prompt])
    st.write(result.content)
