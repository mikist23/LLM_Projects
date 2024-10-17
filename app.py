import os

import streamlit as st
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper



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
    input_variables=["title", 'wikipedia_research'],
    template = "Write me a youtube video script based on this title TITLE: {title}  while leveraging this wikipedia research: {wikipedia_research}"
)




# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')



title_chain = LLMChain(llm = llm, prompt = title_tmplate, verbose =True, output_key = "title", memory = title_memory)
script_chain = LLMChain(llm = llm, prompt = script_tmplate, verbose =True, output_key = "script", memory = script_memory)


# sequential_chain = SequentialChain(chains = [title_chain, script_chain], input_variables = ['topic'],
#                                    output_variables = ['title', 'script'], verbose = True)



if prompt:
    response = sequential_chain({'topic': prompt})
    st.write(response['title'])
    st.write(response['script'])

    with st.expander("Message History"):
        st.info(memory.buffer)