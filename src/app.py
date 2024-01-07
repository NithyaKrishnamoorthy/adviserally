"""
This module serves as a chatbot application using Streamlit. It can address queries from various documents.
"""

import os
import sqlite3
from typing import Any, List, Tuple

import boto3
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import yaml
import json
from bs4 import BeautifulSoup as Soup
from dotenv import load_dotenv

# load agents and tools modules
from langchain.agents import AgentType, Tool, create_sql_agent, initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema.runnable import RunnableConfig
from langchain.sql_database import SQLDatabase
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import DuckDuckGoSearchResults, Tool
from langchain.vectorstores import FAISS
from langchain.vectorstores import Qdrant
from langchain.schema import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever
from loguru import logger
from yaml.loader import SafeLoader

from selfquery_helper import metadata_field_info, document_content_description

# Load environment variables from .env file
load_dotenv(".streamlit/secrets.toml")


def download_sqlite_files():
    """Download SQLite files from S3."""
    # Initialize the S3 client
    s3 = boto3.client("s3")

    # Define the bucket name and the file key
    bucket_name = "for-singlife"
    file_key = "projects/advisorally/my_database.sqlite"
    download_path = "data/sqlite/my_database.sqlite"

    # Download the file
    s3.download_file(bucket_name, file_key, download_path)
    logger.info(f"Downloaded {file_key} from {bucket_name} to {download_path}")


# Set page configuration
st.set_page_config(page_title="AdvisorAlly: Your Digital Guide to Insurance Queries", page_icon="💬")


@st.cache_resource(ttl="1h")
def init_coherent_chain() -> RetrievalQA:
    template = """
    You will offer detailed insights into critical illness plans available in the market, from Manulife, Income, Singlife and HSBC.
    You will provide product comparisons, inlcuding but not limited to the following information:
    Plan Features: Get details about the unique features of different CI plans. For example, ask for plans that offer multiple payouts or cover specific illnesses.
    Coverage Terms: Understand the term options available, whether fixed-term or coverage until a certain age.
    Issue Age Range: Find out the minimum and maximum ages at which you can be issued these plans.
    Currency Options: Discover which plans are available in different currencies like USD, SGD, HKD, GBP, EUR, AUD, etc.
    Premium Terms: Learn about the various premium payment terms and conditions.
    Insurer Information: Identify which insurer offers which plan along with their brochures for more detailed information.
    Effective Dates and Status: Check the effective date of the plans and their current status (e.g., active, draft).
    Do not include insurer icons or images in your answer. 

    If you don't know the answer, just say that you don't know.
    Don't try to make up an answer. 
    ALWAYS return the comparison results in markdown table format. 

    {context}

    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question",
        ],
    )
    df = pd.read_csv("data/excel/coherent_ci_plans_231203.csv", index_col=0)
    docs = []
    for i, row in df.iterrows():
        content = f'{row["insurer.name"]} {row["name"]} {json.dumps(row.to_dict())}'
        # content = f'{json.dumps(row.to_dict())}'
        docs.append(Document(page_content=content, metadata=row.to_dict()))

    store = Qdrant.from_documents(
        docs,
        OpenAIEmbeddings(),
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="my_documents",
    )
    retriever = SelfQueryRetriever.from_llm(
        ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model_name="gpt-4-1106-preview", temperature=0.0),
        store,
        document_content_description,
        metadata_field_info,
        enable_limit=True,
        verbose=False,
        search_kwargs={"k": 4},
    )
    coherent_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model_name="gpt-4-1106-preview", temperature=0.0),
        chain_type="stuff",
        retriever=retriever,
        verbose=False,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )
    return coherent_chain


@st.cache_resource(ttl="1h")
def init_sql_agent() -> AgentExecutor:
    """Initialize SQL agent.

    Returns:
        sql_agent: Initialized SQL agent.
    """
    if not os.path.isfile("./data/sqlite/my_database.sqlite"):
        download_sqlite_files()

    database = SQLDatabase.from_uri("sqlite:///data/sqlite/my_database.sqlite")
    toolkit = SQLDatabaseToolkit(
        db=database,
        llm=ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model_name="gpt-4", temperature=0.0),
    )

    sql_agent = create_sql_agent(
        llm=ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model_name="gpt-4", temperature=0.0),
        toolkit=toolkit,
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    return sql_agent


@st.cache_resource(ttl="1h")
def init_pdf_retriever_from_vectorstore() -> RetrievalQA:
    """Initialize PDF retriever from vector store.

    Returns:
        pdf_chain: Initialized PDF retriever.
    """
    embeddings_path = "./data/embeddings/"
    embeddings_file_name = "pdfs_vectorstore.pkl"

    if not os.path.exists(os.path.join(embeddings_path, embeddings_file_name)):
        loader = PyPDFDirectoryLoader("./data/pdfs/")
        pages = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
        docs, metadatas = [], []
        for page in pages:
            splits = text_splitter.split_text(page.page_content)
            docs.extend(splits)
            metadatas.extend([{"source": page.metadata["source"], "page": page.metadata["page"]}] * len(splits))
            logger.info(f"Split {page.metadata['source']} page {page.metadata['page']} into {len(splits)} chunks")
        store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
        store.save_local(os.path.join(embeddings_path, embeddings_file_name))
    else:
        store = FAISS.load_local(os.path.join(embeddings_path, embeddings_file_name), OpenAIEmbeddings())

    llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model_name="gpt-4", temperature=0.0)

    # initialize vectorstore retriever object
    pdf_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}),
    )
    return pdf_chain


@st.cache_resource(ttl="1h")
def init_pias_pdf_retriever_from_vectorstore() -> RetrievalQA:
    """Initialize PIAS PDF retriever from vector store.

    Returns:
        pdf_chain: Initialized PDF retriever.
    """

    template = """
    Provide comprehensive guidelines for Financial Adviser Representatives (FARs) to organize and participate in public events and activities for client acquisition and sales opportunities, in compliance with the Monetary Authority of Singapore’s standards.
    Provides detailed guidelines on various event types, pre-assessment criteria, good practices, specific event guidelines (such as roadshows, DTD and street canvassing), data protection, creating a conducive environment, incentives and gifts, mystery shopping, post-event surveys, site visits, client call-backs, use of pullers, and engaging introducers.
    If you don't know the answer, just say that you don't know.
    Don't try to make up an answer. 
    Append the source of information: <pdf name, page number>, at the end of your answer.

    {context}

    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question",
        ],
    )
    embeddings_path = "./data/embeddings/"
    embeddings_file_name = "pias_pdfs_vectorstore.pkl"

    if not os.path.exists(os.path.join(embeddings_path, embeddings_file_name)):
        loader = PyPDFDirectoryLoader("./data/pdfs/pias/")
        pages = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, separator="\n")
        docs, metadatas = [], []
        for page in pages:
            splits = text_splitter.split_text(page.page_content)
            docs.extend(splits)
            metadatas.extend([{"source": page.metadata["source"], "page": page.metadata["page"]}] * len(splits))
            logger.info(f"Split {page.metadata['source']} page {page.metadata['page']} into {len(splits)} chunks")
        store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
        store.save_local(os.path.join(embeddings_path, embeddings_file_name))
    else:
        store = FAISS.load_local(os.path.join(embeddings_path, embeddings_file_name), OpenAIEmbeddings())

    llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model_name="gpt-4-1106-preview", temperature=0.0)

    # initialize vectorstore retriever object
    pdf_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return pdf_chain


@st.cache_resource(ttl="1h")
def init_webpages_retriever_from_vectorstore() -> RetrievalQA:
    """Initialize web pages retriever from vector store.

    Returns:
        webpage_chain: Initialized web pages retriever.
    """
    embeddings_path = "./data/embeddings/"
    embeddings_file_name = "webpages_vectorstore.pkl"

    if not os.path.exists(os.path.join(embeddings_path, embeddings_file_name)):
        url = "https://www.moneysmart.sg/cancer-insurance"
        loader = RecursiveUrlLoader(url=url, max_depth=3, extractor=lambda x: Soup(x, "html.parser").text)
        pages = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
        docs, metadatas = [], []
        for page in pages:
            splits = text_splitter.split_text(page.page_content)
            docs.extend(splits)
            metadatas.extend([{"source": page.metadata["source"]}] * len(splits))
            logger.info(f"Split {page.metadata['source']} into {len(splits)} chunks")
        store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
        store.save_local(os.path.join(embeddings_path, embeddings_file_name))
    else:
        store = FAISS.load_local(os.path.join(embeddings_path, embeddings_file_name), OpenAIEmbeddings())

    llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model_name="gpt-4", temperature=0.0)

    # initialize vectorstore retriever object
    webpage_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}),
    )
    return webpage_chain


@st.cache_resource(ttl="1h")
def init_duckduckgo_agent() -> DuckDuckGoSearchResults:
    ddg_agent = DuckDuckGoSearchResults(
        name="Web Search Results",
        backend="api",
        description="""A wrapper around Duck Duck Go Search. Useful for when you need to answer when there are no more alternatives. 
        Input should be a search query. Output is a JSON array of the query results. 
        Please search in Singapore region for best results.
        """,
    )
    return ddg_agent


@st.cache_resource(ttl="1h")
def load_config_file(config_file_path: str) -> dict:
    """Load configuration file.

    Args:
        config_file_path: Path to the configuration file.

    Returns:
        config: Configuration file.
    """
    with open(config_file_path) as file:
        config = yaml.load(file, Loader=SafeLoader)
        logger.info("Config loaded.")
    return config


if __name__ == "__main__":
    home_title = "💬 AdvisorAlly"
    st.markdown(f"""# {home_title} <span style=color:#2dd17a><font size=5>Beta</font></span>""", unsafe_allow_html=True)
    st.info(
        "Advise with confidence. Access policy insights, compare insurance products, and get answers to your general inquiries effortlessly."
    )
    st.markdown(
        """
    <style>
        [data-testid=stSidebar] {
            /* use https://cssgradient.io/ to generate gradient */
            background: rgb(242,241,255);
            background: linear-gradient(0deg, rgba(242,241,255,1) 0%, rgba(242,242,242,1) 44%, rgba(193,233,212,1) 78%);
            color: black;  /* Changing font color to black */
            }
            [data-testid="stSidebar"] .stButton > button {
                color: rgba(45,209,122,1);
            }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # with st.sidebar:
    st.sidebar.image("assets/Advisor Ally Logo.png")
    st.sidebar.markdown(
        """
        **AdvisorAlly** is a GenAI-driven chatbot. This tool equips insurance advisors with
        technology for product evaluation, comparisons, policy insights, and addressing general inquiries. It 
        empowers advisors to make data-driven recommendations and deliver top-tier service to customers.
        
        """
    )
    st.sidebar.warning(
        "Although AdvisorAlly has conversational features, quality of responses will depreciate when context is too long. Remember to **reset chat history** every now and then!"
    )
    st.sidebar.warning(
        "If this app is not working for you, we have likely reached the default rate limit set by OpenAI service. Please give us some time to fix this issue."
    )
    st.sidebar.info("🫶 Built by TOOB, powered by GPT-4.")
    st.sidebar.markdown("---")
    is_clear_btn = st.sidebar.button("Reset chat history")

    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
    )
    # if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    if len(msgs.messages) == 0 or is_clear_btn:
        msgs.clear()
        msgs.add_ai_message("How can I help you?")
        st.session_state.steps = {}

    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            # Render intermediate steps if any were saved
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                    st.write(step[0].log)
                    st.write(step[1])
            st.write(msg.content)

    sql_agent = init_sql_agent()
    pdf_chain = init_pdf_retriever_from_vectorstore()
    pias_pdf_chain = init_pias_pdf_retriever_from_vectorstore()
    webpage_chain = init_webpages_retriever_from_vectorstore()
    coherent_chain = init_coherent_chain()
    ddg_agent = init_duckduckgo_agent()
    logger.info("Agents and retrievers initialized.")
    tools = [
        Tool(
            name="Compare Insurance Plans/Products",
            func=coherent_chain,
            description="""
            Useful for when you need to compare insurance plans/products, e.g. cancer insurance plans.
            When query contains the 'compare' keyword, do not ask what to compare, always compare the all relevant non-empty features of the products in markdown table format.
            For e.g. values like features, coverage term, max coverage age, premium term, effective date, min issue age, max issue age, etc.
            ALWAYS return the comparison results in markdown table format. 
            """,
        ),
        Tool(
            name="Insurance Policies Brochure",
            func=pdf_chain.run,
            description="""
            Useful for when you need to answer questions about insurance products from HSBC. 
            Do not say that it's recommended to seek advise from a financial planner or advisor before making a decision, because the chatbot user is a financial planner. 
            """,
        ),
        Tool(
            name="PIAS Marketing Handbook",
            func=pias_pdf_chain,
            description="""
            Useful for when you need to ask questions about organizing and participating in marketing and client acquisition events for financial advisory services, 
            the 'PIAS Events & Marketing Handbook For FARs' provides detailed guidelines on various event types, pre-assessment criteria, good practices, 
            specific event guidelines (such as roadshows, DTD and street canvassing), data protection, creating a conducive environment, incentives and gifts, 
            mystery shopping, post-event surveys, site visits, client call-backs, use of pullers, and engaging introducers.
            """,
        ),
        Tool(
            name="Term Life Polices",
            func=sql_agent.run,
            description=f"""
            Useful for when you need to answer questions about premium for term life products data stored in sql table term_life_table.
            Table name has no quotes. 
            Run sql operations on table to help you get the right answer.
            """,
        ),
        Tool(
            name="Cancer Insurance Plans 2023",
            func=webpage_chain.run,
            description=f"""
            Useful for when you need to answer questions about cancer insurance plans in singapore too.  
            """,
        ),
        ddg_agent,
    ]
    # change the value of the prefix argument in the initialize_agent function. This will overwrite the default prompt template of the zero shot agent type
    agent_kwargs = {
        "prefix": f"You are friendly financial advisor that advises on insurance polices. You are tasked to assist the current user on questions related to term life policies. You have access to the following tools:"
    }

    llm_agent = initialize_agent(
        tools,
        llm=ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model_name="gpt-4", temperature=0.0),
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        max_iterations=20,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        agent_kwargs=agent_kwargs,
    )
    # llm_agent, msgs = init_llm_agent(sql_agent, pdf_chain, webpage_chain, ddg_agent)
    logger.info("LLM agent initialized.")

    # Process and store input query
    prompt = st.chat_input()

    if prompt:
        st.chat_message("user").write(prompt)
        # Setup LLM agent
        # Initialize agents and retrievers

        # Check if there's a stored response in the session state and render it
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = llm_agent.invoke({"input": prompt}, config=RunnableConfig(callbacks=[st_cb]))["output"]
            response = response.replace("$", "\$")
            st.write(response)
