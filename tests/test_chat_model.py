import unittest
from unittest.mock import patch

from chat_model import ChatModel


class TestChatModel(unittest.TestCase):

    def setUp(self):
        self.model_name = "test_model"
        self.chat_model = ChatModel(self.model_name)

    @patch("chat_model.AutoTokenizer")
    def test_set_tokenizer(self, auto_tokenizer_mock):
        tokenizer_returned_value = auto_tokenizer_mock.from_pretrained.return_value
        self.chat_model.set_tokenizer()
        self.assertIsNotNone(self.chat_model.tokenizer)
        self.assertEqual(tokenizer_returned_value, self.chat_model.tokenizer)
        auto_tokenizer_mock.from_pretrained.assert_called_once()
        auto_tokenizer_mock.from_pretrained.assert_called_once_with(
            self.chat_model.model_name, return_token_type_ids=False
        )

    @patch("chat_model.ChatModel.set_tokenizer")
    def test_get_tokenizer(self, set_tokenizer_mock):
        returned_value = self.chat_model.get_tokenizer()
        self.assertEqual(returned_value, self.chat_model.tokenizer)
        set_tokenizer_mock.assert_called_once()

    @patch("chat_model.os.getenv")
    @patch("chat_model.HuggingFaceEndpoint")
    def test_set_llm(self, hugging_face_endpoint_mock, os_env_mock):
        os_env_returned_value = os_env_mock.return_value
        returned_value = hugging_face_endpoint_mock.return_value
        self.chat_model.set_llm()
        self.assertIsNotNone(self.chat_model.llm)
        self.assertEqual(returned_value, self.chat_model.llm)
        os_env_mock.assert_called_once_with("HUGGINGFACEHUB_API_TOKEN")
        hugging_face_endpoint_mock.assert_called_once_with(
            repo_id=self.model_name,
            max_new_tokens=1000,
            do_sample=True,
            temperature=1.0,
            huggingfacehub_api_token=os_env_returned_value,
            verbose=True,
        )

    @patch("chat_model.ChatModel.set_llm")
    def test_get_llm(self, set_llm_mock):
        returned_value = self.chat_model.get_llm()
        set_llm_mock.assert_called_once()
        self.assertEqual(returned_value, self.chat_model.llm)

    @patch("chat_model.ChatHuggingFace")
    @patch("chat_model.ChatModel.get_llm")
    def test_set_chat_model(self, get_llm_mock, chat_hugging_face_mock):
        self.chat_model.set_chat_model()
        get_llm_mock.assert_called_once()
        chat_hugging_face_mock.assert_called_once_with(
            llm=get_llm_mock.return_value, verbose=True
        )
        self.assertIsNotNone(self.chat_model.chat_model)

    @patch("chat_model.ChatModel.set_chat_model")
    def test_get_chat_model(self, set_chat_model_mock):
        # Test get_chat_model without pre-existing chat_model
        self.assertIsNone(self.chat_model.chat_model)
        result = self.chat_model.get_chat_model()

        set_chat_model_mock.assert_called_once()
        self.assertEqual(result, self.chat_model.chat_model)
