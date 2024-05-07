import unittest
from unittest.mock import Mock, patch

from main import RAGApp  # Assuming your script is named rag_app.py


class StSessionStateMock(Mock, dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


class TestRAGApp(unittest.TestCase):

    @patch("main.st")
    @patch("main.load_dotenv")
    @patch("main.RAGApp.get_model")
    @patch("main.RAGApp.get_indexer")
    @patch("main.RAGApp.get_vectorstore")
    @patch("main.RAGApp.get_retriever")
    @patch("main.RAGApp.get_rag_chain")
    def setUp(
        self,
        get_rag_chain_mock,
        get_retriever_mock,
        get_vectorstore_mock,
        get_indexer_mock,
        get_model_mock,
        load_dotenv_mock,
        st_mock,
    ):
        """Setup test environment before each test."""
        self.mock_state = StSessionStateMock(name="MockStSessionState")
        self.mock_session_id = str(id(self.mock_state))
        self.rag_chain = Mock(name="MockRagChain")
        self.retriever = Mock(name="MockVectorStoreRetriever")
        self.vectorstore = Mock(name="MockVectorStore")
        self.indexer = Mock(name="MockIndexer")
        self.model = Mock(name="MockChatModel")
        st_mock.session_state = self.mock_state
        get_rag_chain_mock.return_value = self.rag_chain
        get_retriever_mock.return_value = self.retriever
        get_vectorstore_mock.return_value = self.vectorstore
        get_indexer_mock.return_value = self.indexer
        get_model_mock.return_value = self.model
        self.app = RAGApp()

    def test_init_sets_up_correct_attributes(self):
        """Test RAGApp initializes correctly with expected attributes."""
        self.assertIsNotNone(self.app.model)
        self.assertIsNotNone(self.app.indexer)
        self.assertIsNotNone(self.app.vectorstore)
        self.assertIsNotNone(self.app.retriever)
        self.assertIsNotNone(self.app.rag_chain)
        self.assertEqual(self.app.model, self.model)
        self.assertEqual(self.app.indexer, self.indexer)
        self.assertEqual(self.app.vectorstore, self.vectorstore)
        self.assertEqual(self.app.retriever, self.retriever)
        self.assertEqual(self.app.rag_chain, self.rag_chain)
        self.assertIsNone(self.mock_state["user_message"])

    @patch("main.ChatModel")
    def test_get_model(self, chat_model_mock):
        chat_model_mock.return_value = self.model
        model = self.app.get_model()
        self.assertIsInstance(
            model, Mock
        )  # Assuming type checking for specific classes
        self.assertTrue(self.mock_state["model"])
        self.assertEqual(model, self.mock_state["model"])
        self.assertEqual(model, self.model)

    @patch("main.Indexer")
    def test_get_indexer(self, indexer_mock):
        indexer_mock.return_value = self.indexer
        indexer = self.app.get_indexer()
        self.assertIsInstance(indexer, Mock)
        self.assertTrue(self.mock_state["indexer"])
        self.assertEqual(indexer, self.app.indexer)
        self.assertEqual(indexer, self.indexer)

    @patch("main.Indexer")
    def test_get_vectorstore(self, indexer_mock):
        indexer_mock.return_value = self.indexer
        self.indexer.get_vectorstore.return_value = self.vectorstore
        vectorstore = self.app.get_vectorstore()
        self.assertIsInstance(vectorstore, Mock)
        self.assertTrue(self.mock_state["vectorstore"])
        self.assertEqual(vectorstore, self.app.vectorstore)
        self.assertEqual(vectorstore, self.vectorstore)

    @patch("main.Retriever")
    @patch("main.RAGApp.get_vectorstore")
    def test_get_retriever(self, get_vectorstore_mock, retriever_mock):
        get_vectorstore_mock.return_value = self.vectorstore
        retriever_mock.return_value = Mock(name="MockRetriever")
        retriever_mock.return_value.get_retriever.return_value = self.retriever
        retriever = self.app.get_retriever()
        self.assertIsInstance(retriever, Mock)
        retriever_mock.assert_called_once_with(self.vectorstore)
        self.assertTrue(self.mock_state["retriever"])
        self.assertEqual(retriever, self.app.retriever)
        self.assertEqual(retriever, self.retriever)

    @patch("main.RagChain")
    @patch("main.RAGApp.get_model")
    @patch("main.RAGApp.get_retriever")
    def test_get_rag_chain(self, get_retriever_mock, get_model_mock, rag_chain_mock):
        get_retriever_mock.return_value = self.retriever
        get_model_mock.return_value = self.model
        rag_chain_mock.return_value = self.rag_chain
        rag_chain = self.app.get_rag_chain()
        self.assertIsInstance(rag_chain, Mock)
        self.assertTrue(self.mock_state["rag_chain"])
        self.assertEqual(rag_chain, self.app.rag_chain)
        self.assertEqual(rag_chain, self.rag_chain)
        rag_chain_mock.assert_called_once_with(
            chat_model=self.model,
            retriever=self.retriever,
            session_id=self.mock_session_id,
        )

    @patch("main.RagChain")
    def test_generate_response(self, rag_chain_mock):
        """Test the response generation based on an input string."""
        mock_input_text = "Hello, test!"
        rag_chain_mock.get_rag_chain.return_value = self.rag_chain
        rag_chain_mock.get_rag_chain.return_value.query.return_value = "Hello, reply!"
        response = self.app.generate_response(mock_input_text)
        self.app.rag_chain.query.assert_called_once_with(mock_input_text)
        self.rag_chain.query.assert_called_once_with(mock_input_text)
        self.assertEqual(response, "Hello, reply!")

    @patch("main.st.file_uploader")
    @patch("main.NamedTemporaryFile")
    @patch("main.os.remove")
    def test_upload_and_index_files(
        self,
        os_remove_mock,
        named_temporary_file_mock,
        st_file_uploader_mock,
    ):
        """
        Test the upload_and_index_files method to see if it correctly handles file uploads and document indexing.
        """
        # Setup mock for the uploaded files
        mock_uploaded_file = Mock(name="MockFile")
        mock_temporary_file = Mock(name="MockTemporaryNamedFile")
        mock_temporary_file.name = "temp_test_document.txt"
        named_temporary_file_mock.return_value.__enter__.return_value = (
            mock_temporary_file
        )
        mock_uploaded_file.name = "test_document.txt"
        mock_uploaded_file.getbuffer.return_value = b"Hello, world!"

        # Mock st.file_uploader to return our mock uploaded files
        st_file_uploader_mock.return_value = [mock_uploaded_file]
        self.indexer.files = []
        self.app.indexer = self.indexer
        # Run the method under test
        self.app.upload_and_index_files()

        # Assert that file_uploader was called with expected arguments
        st_file_uploader_mock.assert_called_with(
            "Choose a file", accept_multiple_files=True
        )
        mock_temporary_file.write.assert_called_once_with(b"Hello, world!")
        # Verify that the document was added to the indexer
        os_remove_mock.assert_called_once_with("temp_test_document.txt")
        self.indexer.add_doc.assert_called_once()
        self.indexer.add_doc.assert_called_once_with(
            "test_document.txt", "temp_test_document.txt"
        )  # Extract positional arguments

    @patch("main.st.chat_input")
    @patch("main.st.chat_message")
    @patch("main.st.spinner")
    @patch("main.RAGApp.generate_response")
    @patch("main.st.write")
    @patch("main.RAGApp.get_rag_chain")
    def test_chat_interface(
        self,
        get_rag_chain_mock,
        write_mock,
        generate_response_mock,
        spinner_mock,
        chat_message_mock,
        chat_input_mock,
    ):
        mock_prompt = "Hello, test!"
        mock_message1 = Mock("MockMessage 1")
        mock_message1.type = "human"
        mock_message1.content = "Hello"
        mock_message2 = Mock("MockMessage 2")
        mock_message2.type = "ai"
        mock_message2.content = "Hi"
        mock_history = Mock("MockHistory")
        mock_history.messages = [mock_message1, mock_message2]
        get_rag_chain_mock.return_value = self.rag_chain
        self.rag_chain.get_session_history.return_value = mock_history
        chat_input_mock.return_value = mock_prompt
        generate_response_mock.return_value = "Hello, reply!"
        self.app.chat_interface()
        self.rag_chain.get_session_history.assert_called_once_with(self.mock_session_id)
        chat_message_mock.assert_any_call("human")
        chat_message_mock.assert_any_call("ai")
        write_mock.assert_any_call("Hello")
        write_mock.assert_any_call("Hi")
        write_mock.assert_any_call("Hello, test!")
        write_mock.assert_any_call("Hello, reply!")
        spinner_mock.assert_called_once_with("Thinking...")

    @patch("main.st.title")
    @patch("main.st.sidebar")
    @patch("main.RAGApp.upload_and_index_files")
    @patch("main.RAGApp.chat_interface")
    def test_run(
        self,
        chat_interface_mock,
        upload_and_index_files_mock,
        st_sidebar_mock,
        st_title_mock,
    ):
        self.app.run()
        chat_interface_mock.assert_called_once()
        st_sidebar_mock.__enter__.assert_called_once()
        st_title_mock.assert_called_once_with("🦜🔗 RAG-exp App")
        upload_and_index_files_mock.assert_called_once()
