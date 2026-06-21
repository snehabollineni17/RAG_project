import os
import logging

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

from ensemble import ensemble_retriever_from_docs, get_model, load_all_pdfs
from memory import create_memory_chain
from rag_chain import make_rag_chain

st.set_page_config(page_title="AUTOSAR RAG Assistant")
st.title("AUTOSAR RAG Assistant")


def show_ui(chain, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chain.invoke(
                    {"question": prompt},
                    config={"configurable": {"session_id": "streamlit-user"}},
                )
                st.markdown(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


@st.cache_resource
def get_chain(google_api_key):
    os.environ["GOOGLE_API_KEY"] = google_api_key
    docs = load_all_pdfs("./data")
    ensemble_retriever = ensemble_retriever_from_docs(docs)
    model = get_model()
    rag_chain = make_rag_chain(model, ensemble_retriever)
    memory_chain = create_memory_chain(model, rag_chain)
    return memory_chain | StrOutputParser()


def get_secret_or_input(secret_key, secret_name, info_link=None):
    if secret_key in st.secrets:
        st.write("Found %s secret" % secret_key)
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value


def run():
    load_dotenv()

    google_api_key = st.session_state.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    with st.sidebar:
        if not google_api_key:
            google_api_key = get_secret_or_input(
                "GOOGLE_API_KEY",
                "Google API key",
                info_link="https://aistudio.google.com/apikey",
            )

    if not google_api_key:
        st.warning("Missing GOOGLE_API_KEY")
        st.stop()

    try:
        chain = get_chain(google_api_key)
        st.subheader("Ask me questions about AUTOSAR?")
        show_ui(chain, "What would you like to know?")
    except Exception as exc:
        logging.exception("Failed to initialize RAG chain")
        st.error("Failed to initialize RAG chain")
        st.exception(exc)
        st.stop()


if __name__ == "__main__":
    run()
