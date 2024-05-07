import os
import uuid
from tempfile import NamedTemporaryFile

import streamlit as st
from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from chat_model import ChatModel
from index import Indexer
from rag import RagChain
from retriever import Retriever


class RAGApp:
    def __init__(self) -> None:
        """
        Initializes the RAG-exp application, setting up all the necessary components and session states

        Returns:
            None:
        """
        load_dotenv()

        # Initialize components

        self.state = st.session_state
        self.session_id = str(id(self.state))
        self.model = self.get_model()
        self.indexer = self.get_indexer()
        self.vectorstore = self.get_vectorstore()
        self.retriever = self.get_retriever()
        self.rag_chain = self.get_rag_chain()
        self.rag_chain.set_rag_chain()
        self.state.user_message = None

    def get_model(self) -> ChatModel:
        """
        Fetches the chat model from session state or initiates one if it does not exist.

        Returns:
            ChatModel:  An instance of the ChatModel used for generating responses.

        """
        if "model" not in self.state.keys():
            self.model = ChatModel(model_name="mistralai/Mistral-7B-Instruct-v0.2")
            self.state["model"] = self.model
        return self.state["model"]

    def get_indexer(self) -> Indexer:
        """
        Retrieves the document indexer from session state or creates one.

        Returns:
            Indexer: An instance of the Indexer used for managing document indexing.

        """
        if "indexer" not in self.state.keys():
            self.indexer = Indexer()
            self.state["indexer"] = self.indexer
        return self.state["indexer"]

    def get_vectorstore(self) -> VectorStore:
        """
        Obtains the vector store from session state, initializing it via the indexer if necessary.

        Returns:
            VectorStore: The vector store handling the embeddings and vector operations.

        """
        if "vectorstore" not in self.state.keys():
            self.vectorstore = self.get_indexer().get_vectorstore()
            self.state["vectorstore"] = self.vectorstore
        return self.state["vectorstore"]

    def get_retriever(self) -> VectorStoreRetriever:
        """
        Acquires the retriever component, setting it up with the vector store if not already present.

        Returns:
            VectorStoreRetriever: The component used for retrieving relevant documents based on queries.

        """
        if "retriever" not in self.state.keys():
            self.retriever = Retriever(self.get_vectorstore()).get_retriever()
            self.state["retriever"] = self.retriever
        return self.state["retriever"]

    def get_rag_chain(self) -> RagChain:
        """
        Fetches or configures the RagChain combining retrieval and generation capabilities.

        Returns:
            RagChain: The component responsible for generating responses using retrieved documents.

        """
        if "rag_chain" not in self.state.keys():
            self.rag_chain = RagChain(
                chat_model=self.get_model(),
                retriever=self.get_retriever(),
                session_id=self.session_id,
            )
            self.state["rag_chain"] = self.rag_chain
        return self.state["rag_chain"]

    def upload_and_index_files(self) -> None:
        """Handles the uploading and indexing of files.

        Returns:
            None:
        """
        uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in self.indexer.files:
                    suffix = uploaded_file.name.split(".")[-1]
                    with NamedTemporaryFile(suffix=f".{suffix}", delete=False) as f:
                        f.write(uploaded_file.getbuffer())
                        self.indexer.add_doc(uploaded_file.name, f.name)
                    os.remove(f.name)

    def generate_response(self, input_text: str) -> str:
        """Generates a response for the given input text using the RAG chain.

        Args:
            input_text (str): The user input text to respond to.

        Returns:
            str: The generated response from the model.
        """
        response = self.rag_chain.query(input_text)
        return response

    def chat_interface(self) -> None:
        """
        Creates and manages the chat interface for the application.

        Returns:
            None:

        """
        if prompt := st.chat_input(
            "Your question"
        ):  # Prompt for user input and save to chat history
            self.state.user_message = prompt
        for message in (
            self.get_rag_chain().get_session_history(self.session_id).messages
        ):
            # Display the prior chat messages
            with st.chat_message(message.type):
                st.write(message.content)

        if self.state.user_message:
            with st.chat_message("human"):
                st.write(self.state.user_message)
            with st.chat_message("ai"):
                with st.spinner("Thinking..."):
                    response = self.generate_response(self.state.user_message)
                    st.write(response)
            self.state.user_message = None

    def run(self) -> None:
        """
        Runs the Streamlit application interface.

        Returns:
            None:
        """
        st.title("🦜🔗 RAG-exp App")

        with st.sidebar:
            self.upload_and_index_files()

        self.chat_interface()


if __name__ == "__main__":
    app = RAGApp()
    app.run()
