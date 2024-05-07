from operator import itemgetter
from typing import Dict, List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import ConfigurableFieldSpec, RunnablePassthrough
from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStoreRetriever

from chat_model import ChatModel

# Default chat prompt setup for conversation interactions.
DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to "
            "answer the question. If you don't know the answer, just say that you don't know. Cite the sources or "
            "context through which you are answering the question. You are free to answer the questions of user that "
            "pertains to chatbot-domain like greetings, salutation etc., for these questions or answers don't use "
            "the context or provide the sources use your own ability as assistant. Keep your answer concise and stick "
            "to the question and if required to the sources and again if required to the chat history."
            "Context: {context} \n Chat History: {history} \n Question: {question}"
            "\n Answer:",
        ),
    ]
)


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store

        Args:
            message (BaseMessage):

        Returns:
            None: Returns NoneType object
        """
        self.messages.append(message)

    def clear(self) -> None:
        """
        Clears all messages stored in the history.

        Returns:
            None: Returns NoneType object

        """
        self.messages = []


class RagChain:
    def __init__(
        self,
        retriever: VectorStoreRetriever,
        chat_model: ChatModel,
        session_id: str,
        prompt: ChatPromptTemplate = DEFAULT_PROMPT,
    ) -> None:
        """
        Initializes the RagChain with necessary components.

        Args:
            retriever (VectorStoreRetriever): The component for retrieving relevant documents based on input queries.
            chat_model (ChatModel): The chat model used for processing and generating chat responses.
            session_id (str): The chat session id which will be used to track memory.
            prompt (ChatPromptTemplate): The prompt template to use for generating chat prompts.

        Returns:
            None: Returns NoneType object
        """
        self.rag_chain = None
        self.retriever = retriever
        self.chat_model = chat_model
        self.prompt = prompt
        self.store: Dict[str, BaseChatMessageHistory] = {}
        self.session_id = session_id

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Retrieves or initializes the chat history for a given session ID.

        Args:
            session_id (str): The unique identifier for the session.

        Returns:
            BaseChatMessageHistory: The chat history associated with the session ID.

        """
        if session_id not in self.store:
            self.store[session_id] = InMemoryHistory()
        return self.store[session_id]

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """
        Formats a list of retrieved documents into a single string with separated entries.

        Args:
            docs (List[Document]): A list of Document objects.

        Returns:
            str: A formatted string of document content suitable for including in a prompt.

        """
        return "\n\n".join(doc.page_content for doc in docs)

    def set_rag_chain(self) -> None:
        """
        Sets up the full retrieval-augmentation-generation chain for handling chat queries.

        Returns:
            None: Returns object of NoneType
        """
        context = itemgetter("question") | self.retriever | self.format_docs
        first_step = RunnablePassthrough.assign(context=context)
        rag_chain = (
            first_step
            | self.prompt
            | self.chat_model.get_chat_model()
            | StrOutputParser()
        )

        self.rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history=self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    name="Session ID",
                    description="Unique identifier for the session.",
                    default="",
                    is_shared=True,
                ),
            ],
            verbose=True,
        )

    def get_rag_chain(self) -> RunnableBindingBase:
        """
        Retrieves or initializes the configured RAG chain for query processing.

        Returns:
            RunnableBindingBase: The fully configured and operational RAG chain.

        """
        if self.rag_chain is None:
            self.set_rag_chain()
        return self.rag_chain

    def query(self, text: str) -> str:
        """
        Processes an input text query through the RAG chain and returns a response.

        Args:
            text (str): The input query text to process.

        Returns:
            str: The generated response based on the input text and retrieved context.

        """
        return self.get_rag_chain().invoke(
            {"question": text}, config={"configurable": {"session_id": self.session_id}}
        )
