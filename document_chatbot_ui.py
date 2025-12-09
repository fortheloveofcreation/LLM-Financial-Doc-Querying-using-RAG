import os

import streamlit as st
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

from streamlit.logger import get_logger

logger = get_logger(__name__)

# Log full text sent to LLM
VERBOSE = False

# Details of persisted embedding store index
COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Size of window for buffered window memory
MEMORY_WINDOW_SIZE = 10

ANSWER_ROLE = "Chatbot"
FIRST_MESSAGE = "How can I help you?"
QUESTION_ROLE = "User"
PLACE_HOLDER = "Your message"


# Cached shared objects
@st.cache_resource
def load_embeddings():
    embeds = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return embeds


@st.cache_resource
def load_llm():
    llm = Ollama(model="llama3.1", temperature=0)
    return llm


@st.cache_resource
def get_embed_retriever():
    db = get_embed_db(embeddings)
    retriever = db.as_retriever()
    return retriever


def get_embed_db(embeddings):
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    db = get_chroma_db(embeddings, chroma_persist_dir)
    return db


def get_chroma_db(embeddings, persist_dir):
    db = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    return db


# Shared/cached globals
embeddings = load_embeddings()
llm = load_llm()
retriever = get_embed_retriever()


def save_message(role, content, sources=None):
    logger.info(f"message: {role} - '{content}'")
    msg = {"role": role, "content": content, "sources": sources}
    st.session_state["messages"].append(msg)
    return msg


def source_description(md):
    descr = f"{md['source']}, Page {md['page']}"
    return descr


def write_message(msg):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["sources"]:
            st.write(", ".join([source_description(md) for md in msg["sources"]]))


st.title("Document Chatbot")

st.write(
    """This conversational interface allows you to interact with
indexed content, in this case, The Federalist Papers."""
)

if "query_chain" not in st.session_state:
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        window_size=MEMORY_WINDOW_SIZE,
    )
    st.session_state["query_chain"] = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        verbose=VERBOSE,
        return_source_documents=True,
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    save_message(ANSWER_ROLE, FIRST_MESSAGE)

for msg in st.session_state["messages"]:
    write_message(msg)

if prompt := st.chat_input(PLACE_HOLDER):
    msg = save_message(QUESTION_ROLE, prompt)
    write_message(msg)

    qa = st.session_state["query_chain"]
    query_response = qa({"question": prompt})
    response = query_response["answer"]
    source_docs = [d.metadata for d in query_response["source_documents"]]
    msg = save_message(ANSWER_ROLE, response, source_docs)
    write_message(msg)
