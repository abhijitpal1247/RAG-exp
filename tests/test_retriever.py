import unittest
from unittest.mock import Mock, patch

from retriever import Retriever


class TestRetriever(unittest.TestCase):
    def setUp(self):
        # Mock the vector store dependency
        self.vector_store_mock = Mock(name="MockVectorStore")
        self.vector_store_mock.as_retriever = Mock(
            return_value=Mock(name="MockRetriever")
        )

        # Instantiate the Retriever with the mocked vectorstore
        self.retriever = Retriever(vectorstore=self.vector_store_mock)

    def test_set_retriever(self):
        new_search_kwargs = {"k": 10, "metric": "euclidean"}
        new_search_type = "custom_search"

        self.retriever.set_retriever(
            search_type=new_search_type, search_kwargs=new_search_kwargs
        )

        self.assertEqual(self.retriever.search_type, new_search_type)
        self.assertEqual(self.retriever.search_kwargs, new_search_kwargs)
        self.vector_store_mock.as_retriever.assert_called_once_with(
            search_type=new_search_type, search_kwargs=new_search_kwargs
        )
        self.assertIsNotNone(self.retriever.retriever)

    @patch("retriever.Retriever.set_retriever")
    def test_get_retriever(self, set_retriever_mock):
        # Ensuring that retriever is set up lazily
        self.assertIsNone(self.retriever.retriever)
        retriever_instance = self.retriever.get_retriever()

        set_retriever_mock.assert_called_once()
        self.assertEqual(retriever_instance, self.retriever.retriever)

    @patch("retriever.Retriever.get_retriever")
    def test_retrieve_docs(self, get_retriever_mock):
        dummy_query = "what is unit testing?"
        dummy_response = ["Document 1", "Document 2"]
        get_retriever_mock.return_value.invoke.return_value = dummy_response

        result = self.retriever.retrieve_docs(dummy_query)

        get_retriever_mock.return_value.invoke.assert_called_once_with(dummy_query)
        self.assertEqual(result, dummy_response)
