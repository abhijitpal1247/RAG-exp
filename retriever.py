from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever


class Retriever:
    def __init__(
        self,
        vectorstore: VectorStore,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[Any, Any]] = None,
    ) -> None:
        """
        Initializes a Retriever instance with a vector store and search configurations.

        Args:
            vectorstore (VectorStore): The vector store to be used for document retrieval.
            search_type (str): The type of search to be conducted (e.g., 'similarity', 'mmr').
            search_kwargs (Optional[Dict[Any, Any]]): Additional keyword arguments to influence the search behavior.

        Returns:
            None: Returns object of NoneType
        """
        self.retriever = None
        if search_kwargs is None:
            self.search_kwargs = {"k": 6}
        self.search_type = search_type
        self.vectorstore = vectorstore

    def set_retriever(
        self,
        search_type: Optional[str] = None,
        search_kwargs: Optional[Dict[Any, Any]] = None,
    ) -> None:
        """
        Configures or reconfigures the retriever settings for the vector store.

        Args:
            search_type (str): Optionally update the search type.
            search_kwargs (Dict[Any, Any]): Optionally update additional search parameters.

        Returns:
            None: Returns object of NoneType
        """
        if search_type is not None:
            self.search_type = search_type
        if search_kwargs is not None:
            self.search_kwargs = search_kwargs
        self.retriever = self.vectorstore.as_retriever(
            search_type=self.search_type, search_kwargs=self.search_kwargs
        )

    def get_retriever(self) -> VectorStoreRetriever:
        """
        Retrieves or initializes the retriever object based on current configuration.

        Returns:
            VectorStoreRetriever: The retriever object ready to be used for document queries.

        """
        if self.retriever is None:
            self.set_retriever()
        return self.retriever

    def retrieve_docs(self, query: str) -> List[Document]:
        """
        Retrieves documents based on a given query string using the configured retriever.

        Args:
            query (str): The search query to retrieve relevant documents.

        Returns:
            List[Document]: A list of Document objects that are relevant to the query.

        """
        return self.get_retriever().invoke(query)
