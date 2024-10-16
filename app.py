import os

import streamlit as st
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain



# App framework
st.title("LANG 🦜🕸️ CHAIN PROMPT")
prompt = st.text_input("Plug in your prompt here")

llm = HuggingFaceHub(
    repo_id = "bigscience/bloom-1b7",
    # repo_id='tiiuae/falcon-7b-instruct',
    # model_kwargs = {'temperature':1e-10},
    
    
)
# prompt templates
title_tmplate = PromptTemplate(
    input_variables=["topic"],
    template = "Write me a youtube video title about {topic}"
)

script_tmplate = PromptTemplate(
    input_variables=["title"],
    template = "Write me a youtube video script based on this title TITLE: {title}"
)


title_chain = LLMChain(llm = llm, prompt = title_tmplate, verbose =True, output_key = "title")
script_chain = LLMChain(llm = llm, prompt = script_tmplate, verbose =True, output_key = "script")

sequential_chain = SequentialChain(chains = [title_chain, script_chain], input_variables = ['topic'],
                                   output_variables = ['title', 'script'], verbose = True)

if prompt:
    response = sequential_chain({'topic': prompt})
    st.write(response['title'])
    st.write(response['script'])