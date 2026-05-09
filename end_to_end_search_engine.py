#### Search Engine with Tools and Agents

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper , WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun , DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.agents import initialize_agent , AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv


## Arxiv and wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results = 5 , doc_context_chars_max = 3000)
arxiv =ArxivQueryRun(api_wrapper = arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results = 5 , doc_content_chars_max = 3000)
wiki = WikipediaQueryRun(api_wrapper = wiki_wrapper)

duck_duck_search = DuckDuckGoSearchRun(
    name = "Search the web",
    description = "Useful search engine for finding the current events and news.",
    handle_tool_error = True
)

st.title("End to End Search Engine with Tools and Agents using GROQ API and Langchain")
"""
In this example , we're using 'StreamlitCallbackHandler' to display the thoughts and actions of the agent in real time on the Streamlit app. The agent is initialized with the Groq API and the tools (ArxivQueryRun and WikipediaQueryRun) that it can use to find information. The agent will decide which tool to use based on the user's query and will display its thought process and actions in the Streamlit app.
"""


# sidebar
# st.sidebar.title("Settings")
# api_key = st.sidebar.text_input("Enter your GROQ API KEY" , type = "password")


load_dotenv()                              # ← moved here
api_key = os.getenv("GROQ_API_KEY") 

if not api_key:
    st.error("GROQ_API_KEY not found in .env file!")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role" : "Assistant" , "content" : "Heyy there Boss , How can I help you today with?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])



# creating prompt
if prompt := st.chat_input(placeholder = "Hello Lord , Drop your query here..."):
    st.session_state.messages.append({'role' : 'user' , 'content' : prompt})
    st.chat_message('user').write(prompt)


    llm = ChatGroq(model = "llama-3.3-70b-versatile" , api_key = api_key , streaming = True)
    tools = [arxiv , wiki , duck_duck_search]

    search_agent = initialize_agent(tools , llm , agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION , verbose = True , handle_parsing_errors = True , max_iterations = 5 , max_execution_time = 60 , early_stopping_method = "generate" , return_trace = True)

    with st.chat_message('Assistant'):
        st_callback = StreamlitCallbackHandler(st.container() , expand_new_thoughts = False)
        response = search_agent.run(st.session_state.messages[-1]['content'] , callbacks = [st_callback])
        st.session_state.messages.append({'role' : 'Assistant' , 'content' : response})
        st.write(response)
