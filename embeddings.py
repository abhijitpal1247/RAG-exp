import os
from typing import Union

import langchain_core
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.pydantic_v1 import BaseModel


class Embeddings:
    def __init__(
        self, embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    ) -> None:
        """
        Initializes the Embeddings class by specifying the model to be used for generating embeddings.

        Args:
            embedding_model_name (str): The name of the model hosted on Hugging Face's model hub used for embeddings.

        Returns:
            None: Returns object of NoneType
        """
        self.embeddings = None
        self.embedding_model_name = embedding_model_name

    def set_embeddings_model(self) -> None:
        """
        Configures and sets the embeddings object using the specified transformer model from Hugging Face API.
        It uses an API key stored in the environment to authenticate on Hugging Face Hub.

        Returns:
            None: Returns object of NoneType

        """
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            model_name=self.embedding_model_name,
            api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )

    def get_embeddings_model(
        self,
    ) -> Union[BaseModel, langchain_core.embeddings.Embeddings]:
        """
        Retrieves the embeddings model object if it has been set, otherwise initializes it using the
        set_embeddings_model method.

        Returns:
            Union[BaseModel, langchain_core.embeddings.Embeddings]: The embeddings model object used for
            generating text embeddings.

        """
        if self.embeddings is None:
            self.set_embeddings_model()
        return self.embeddings
