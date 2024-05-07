import unittest
from unittest.mock import AsyncMock, patch

from index import Indexer


class TestIndexer(unittest.TestCase):
    def setUp(self):
        self.indexer = Indexer()

    @patch("index.PyPDFLoader")
    @patch("index.RecursiveCharacterTextSplitter")
    def test_load_and_split_data(
        self, recursive_character_text_splitter_mock, py_pdf_loader_mock
    ):
        test_file = "test.pdf"
        loader = py_pdf_loader_mock.return_value
        loader.load_and_split.return_value = ["mocked pages"]
        text_splitter = recursive_character_text_splitter_mock.return_value
        text_splitter.split_documents.return_value = ["mocked_splits"]
        splits = self.indexer.load_and_split_data(test_file)

        py_pdf_loader_mock.assert_called_once_with("test.pdf")
        recursive_character_text_splitter_mock.assert_called_once_with(
            chunk_size=1000, chunk_overlap=200
        )
        text_splitter.split_documents.assert_called_once_with(["mocked pages"])
        self.assertEqual(splits, ["mocked_splits"])

    @patch("index.Indexer.load_and_split_data")
    @patch("index.Indexer.get_vectorstore")
    def test_add_doc(self, get_vectorstore_mock, load_and_split_data_mock):
        test_file = "temp_test.pdf"
        test_file_name = "test.pdf"
        vectorstore_mock = get_vectorstore_mock.return_value
        load_and_split_data_mock.return_value = ["mocked stuff"]
        vectorstore_mock.aadd_documents = AsyncMock()
        self.indexer.add_doc(test_file_name, test_file)
        self.assertIn("test.pdf", self.indexer.files)
        load_and_split_data_mock.assert_called_once()
        vectorstore_mock.aadd_documents.assert_called_with(["mocked stuff"])
        vectorstore_mock.aadd_documents.assert_called_once()

    @patch("index.Database")
    @patch("index.Embeddings")
    def test_set_vectorstore(self, embeddings_mock, database_mock):
        vectorstore = database_mock.return_value.get_db.return_value
        vectorstore_embedding = (
            embeddings_mock.return_value.get_embeddings_model.return_value
        )

        self.indexer.set_vectorstore()

        self.assertIsNotNone(self.indexer.vectorstore)
        self.assertEqual(self.indexer.vectorstore, vectorstore)
        self.assertEqual(self.indexer.vectorstore._embedding, vectorstore_embedding)

    @patch("index.Indexer.set_vectorstore")
    def test_get_vectorstore(self, set_vectorstore_mock):
        returned_vectorstore = self.indexer.get_vectorstore()
        set_vectorstore_mock.assert_called_once()
        self.assertEqual(returned_vectorstore, self.indexer.vectorstore)
