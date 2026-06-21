from collections import defaultdict
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


def create_memory_chain(llm, rag_chain):

    contextualize_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given chat history and the latest question, rewrite it as a standalone question. "
            "Do NOT answer it."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    runnable = {
        "question": contextualize_chain
    } | rag_chain

    memory_store = defaultdict(InMemoryChatMessageHistory)

    def get_session_history(session_id: str):
        return memory_store[session_id]

    return RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )