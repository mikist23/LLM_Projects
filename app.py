import os

import streamlit as st
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



# App framework
st.title("LANG 🦜🕸️ CHAIN PROMPT")
prompt = st.text_input("Plug in your prompt here")

llm = HuggingFaceHub(
    repo_id = "bigscience/bloom-1b7",
    # repo_id='tiiuae/falcon-7b-instruct',
    model_kwargs = {'temperature':1e-10},
    
    
)
# prompt templates
title_tmplate = PromptTemplate(
    input_variables=["topic"],
    template = "Write me a youtube video title about {topic}"
)

title_chain = LLMChain(llm = llm, prompt = title_tmplate, verbose =True)


if prompt:
    response = title_chain.run(topic = prompt)
    st.write(response)