import os
from threading import Lock
from typing import Any, Dict, List, Optional

import weaviate
from langchain_core.vectorstores import VectorStore
from langchain_weaviate import WeaviateVectorStore


class DatabaseSingletonMeta(type):
    _instances: Dict[Any, Any] = {}
    _lock: Lock = Lock()

    def __call__(
        cls, *args: Optional[List[Any]], **kwargs: Optional[Dict[Any, Any]]
    ) -> object:
        """
        Initializes the Database object by connecting to a Weaviate cluster and authentication using
        environmental variables.

        Args:
            *args (Optional[List[Any]]): Arbitrarily positional arguments passed to the class constructors.
            **kwargs (Optional[Dict[Any, Any]]): Arbitrary keyword arguments passed to the class constructors.

        Returns:
            object: A singleton instance of the class.

        """
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Database(metaclass=DatabaseSingletonMeta):
    def __init__(self) -> None:
        """
        Initializes the Database object by connecting to a Weaviate cluster and authentication using
        environmental variables.

        Returns:
            None: Returns object of NoneType

        """
        self.db = None
        self.client = weaviate.connect_to_wcs(
            cluster_url=os.getenv("WCS_DEMO_URL"),  # Replace with your WCS URL
            auth_credentials=weaviate.auth.AuthApiKey(
                os.getenv("WCS_DEMO_RO_KEY")
            ),  # Replace with your WCS key
        )

    def set_db(self) -> None:
        """
        Initializes and sets the database variable, setting up the Weaviate vector storage integration.

        Returns:
            None: Returns object of NoneType

        """
        self.db = WeaviateVectorStore(
            client=self.client, index_name="MyIndex", text_key="text"
        )

    def get_db(self) -> VectorStore:
        """
        Retrieves the database object, initializing it if not already done. This ensures lazy initialization.

        Returns:
            VectorStore: The initialized vector store instance as part of this database.

        """
        if self.db is None:
            self.set_db()
        return self.db

    def __del__(self) -> None:
        """
        Safely closes the client connection when the Database object is deleted.

        Returns:
            None: Returns of object of NoneType

        """
        self.client.close()
