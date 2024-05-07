import asyncio
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database_utils import Database
from embeddings import Embeddings


class Indexer:
    def __init__(self) -> None:
        """
        Initializes an Indexer object that tracks the vector store and a list of processed files.

        Returns:
            None: Returns object of NoneType

        """
        self.vectorstore = None
        self.files: List[str] = []

    @staticmethod
    def load_and_split_data(file: str) -> List[Document]:
        """
         Loads a PDF file, splits it into pages, and further splits each page into chunks using defined settings.

        Args:
            file (str): The path to the PDF file to be processed.

        Returns:
            List[Document]: A list of Document objects that represent chunks of text from the file.

        """
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(pages)
        return splits

    def add_doc(self, file_name: str, file: str) -> None:
        """
        Processes a document file, splits it, and adds it to the vector store if not already added.

        Args:
            file_name (str): The name of the file to be processed.
            file (str): The path to the file.

        Returns:
            None:
        """
        if file_name not in self.files:
            self.files.append(file_name)
            vectorstore = self.get_vectorstore()
            chunks = self.load_and_split_data(file)
            asyncio.run(vectorstore.aadd_documents(chunks))

    def set_vectorstore(self) -> None:
        """
        Sets up the vector store by retrieving a database instance and setting an embeddings model for document encoding.

        Returns:
            None: Returns object of NoneType

        """
        self.vectorstore = Database().get_db()
        if self.vectorstore is not None:
            self.vectorstore._embedding = Embeddings().get_embeddings_model()

    def get_vectorstore(self) -> VectorStore:
        """
        Retrieves the vector store, setting it up if it hasn't been set already.

        Returns:
            VectorStore: The vector store used for storing document vectors.

        """
        if self.vectorstore is None:
            self.set_vectorstore()
        return self.vectorstore
