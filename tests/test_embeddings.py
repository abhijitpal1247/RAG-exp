import unittest
from unittest.mock import patch

from embeddings import (
    Embeddings,  # Import the module containing your class (replace 'your_module')
)


class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        # Create instance of Embeddings with default parameter
        self.embeddings = Embeddings()

    @patch("embeddings.os.getenv")
    @patch("embeddings.HuggingFaceInferenceAPIEmbeddings")
    def test_set_embeddings_model(
        self, hugging_face_inference_api_embedding_class_mock, os_getenv_mock
    ):
        # Mocking the environment variable fetching and the HuggingFaceInferenceAPIEmbeddings constructor
        os_getenv_mock.return_value = "fake_api_token"

        self.assertIsNone(self.embeddings.embeddings)

        self.embeddings.set_embeddings_model()

        # Check if getenv is called correctly
        os_getenv_mock.assert_called_once_with("HUGGINGFACEHUB_API_TOKEN")

        # Ensure our embedding class is initialized correctly with the mock token
        hugging_face_inference_api_embedding_class_mock.assert_called_once_with(
            model_name="sentence-transformers/all-mpnet-base-v2",
            api_key="fake_api_token",
        )

        # Ensure the embeddings instance is not None after setting it
        self.assertIsNotNone(self.embeddings.embeddings)

    @patch("embeddings.Embeddings.set_embeddings_model")
    def test_get_embeddings_model(self, set_embeddings_model_mock):
        self.assertIsNone(self.embeddings.embeddings)
        result = self.embeddings.get_embeddings_model()

        set_embeddings_model_mock.assert_called_once()
        self.assertEqual(result, self.embeddings.embeddings)
