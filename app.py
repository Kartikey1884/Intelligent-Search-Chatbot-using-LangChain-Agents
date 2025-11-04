import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

load_dotenv()

st.title("🔎 LangChain - Chat with Search")

api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Define tools
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200))
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))
search = DuckDuckGoSearchRun(name="Search")

tools = {"wikipedia": wiki, "arxiv": arxiv, "search": search}

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", streaming=True)
    st_cb = StreamlitCallbackHandler(st.container())

    # Simple reasoning: decide which tool to call
    if "arxiv" in prompt.lower():
        response = tools["arxiv"].invoke(prompt)
    elif "wikipedia" in prompt.lower() or "who is" in prompt.lower():
        response = tools["wikipedia"].invoke(prompt)
    elif "search" in prompt.lower() or "find" in prompt.lower():
        response = tools["search"].invoke(prompt)
    else:
        response = llm.invoke(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
