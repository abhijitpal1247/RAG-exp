import unittest
from unittest.mock import Mock, patch

from langchain_core.runnables import ConfigurableFieldSpec

from rag import InMemoryHistory, RagChain  # Update import with your module


class TestRagChain(unittest.TestCase):
    def setUp(self):
        self.retriever = Mock(name="MockRetriever")
        self.chat_model = Mock(name="MockChatModel")
        self.session_id = "test_session_id"
        self.rag_chain = RagChain(
            retriever=self.retriever,
            chat_model=self.chat_model,
            session_id=self.session_id,
        )
        self.rag_chain.prompt = Mock(name="MockPrompt")

    def test_get_session_history_new_session(self):
        history = self.rag_chain.get_session_history(self.session_id)
        self.assertIsInstance(history, InMemoryHistory)
        self.assertEqual(history.messages, [])
        self.assertIn(self.session_id, self.rag_chain.store)

    def test_get_session_history_existing_session(self):
        expected_history = InMemoryHistory()
        expected_history.add_message(Mock())
        self.rag_chain.store[self.session_id] = expected_history

        history = self.rag_chain.get_session_history(self.session_id)
        self.assertEqual(history, expected_history)

    @patch("rag.RunnableWithMessageHistory")
    @patch("rag.RunnablePassthrough.assign")
    @patch("rag.StrOutputParser")
    @patch("rag.itemgetter")
    @patch("rag.RagChain.format_docs")
    @patch("rag.RagChain.get_session_history")
    def test_set_rag_chain(
        self,
        get_session_history_mock,
        format_docs_mock,
        itemgetter_mock,
        str_output_parser_mock,
        runnable_passthrough_assign_mock,
        runnable_with_message_history_mock,
    ):
        self.rag_chain.set_rag_chain()

        itemgetter_mock.assert_called_once_with("question")
        context_mock = (
            itemgetter_mock.return_value | self.rag_chain.retriever | format_docs_mock
        )
        first_step_mock = runnable_passthrough_assign_mock.return_value
        rag_chain_mock = (
            first_step_mock
            | self.rag_chain.prompt
            | self.rag_chain.chat_model
            | str_output_parser_mock
        )
        runnable_passthrough_assign_mock.assert_called_once_with(context=context_mock)
        runnable_with_message_history_mock.assert_called_once_with(
            rag_chain_mock,
            get_session_history=get_session_history_mock,
            input_messages_key="question",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id=self.session_id,
                    annotation=str,
                    name="Session ID",
                    description="Unique identifier for the " "session.",
                    default="",
                    is_shared=True,
                ),
            ],
            verbose=True,
        )

    @patch("rag.RagChain.set_rag_chain")
    def test_get_rag_chain(self, set_rag_chain_mock):
        self.assertIsNone(self.rag_chain.rag_chain)
        result = self.rag_chain.get_rag_chain()
        set_rag_chain_mock.assert_called_once()
        self.assertEqual(result, self.rag_chain.rag_chain)
        # Check if subsequent calls return the same instance
        result_again = self.rag_chain.get_rag_chain()
        self.assertEqual(result, result_again)

    @patch("rag.RagChain.get_rag_chain")
    def test_query(self, get_rag_chain_mock):
        question = "What is unit testing?"
        expected_response = "It is a testing method."
        get_rag_chain_mock.return_value = Mock(name="rag_chain")
        get_rag_chain_mock.return_value.invoke.return_value = expected_response
        response = self.rag_chain.query(question)

        get_rag_chain_mock.return_value.invoke.assert_called_once_with(
            {"question": question},
            config={"configurable": {"session_id": self.session_id}},
        )
        self.assertEqual(response, expected_response)
