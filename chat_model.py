import os

from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.language_models import LLM, BaseChatModel, BaseLLM
from transformers import AutoTokenizer


class ChatModel:
    def __init__(self, model_name: str) -> None:
        """
        Initializes the ChatModel with a given model name.

        Args:
            model_name (str):The model name to use for setting up the tokenizer and the llm.

        Returns:
            None: Returns NoneType
        """
        self.chat_model = None
        self.model_name = model_name
        self.tokenizer = None
        self.llm = None

    def set_tokenizer(self) -> None:
        """
        Sets up the tokenizer based on the pre-trained model specified in the `model_name` parameter during
        initialization.
        Returns:
            None: Returns NoneType object

        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, return_token_type_ids=False
        )

    def get_tokenizer(self) -> AutoTokenizer:
        """
        Retrieves or lazily initializes the tokenizer.

        Returns:
            AutoTokenizer: The tokenizer for the specified llm.

        """
        if self.tokenizer is None:
            self.set_tokenizer()
        return self.tokenizer

    def set_llm(self) -> None:
        """
        Sets up the large language model (LLM) endpoint specific to the chat model needs.

        Returns:
            None: Returns object of NoneType

        """
        repo_id = self.model_name
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_new_tokens=1000,
            do_sample=True,
            temperature=1.0,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            verbose=True,
        )

    def get_llm(self) -> LLM:
        """
        Retrieves or constructs the language model endpoint (LLM).

        Returns:
            BaseLLM: The initialized HuggingFace language model endpoint.

        """
        if self.llm is None:
            self.set_llm()
        return self.llm

    def set_chat_model(self) -> None:
        """
        Sets up the actual chat model integrating the LLM for generating chat responses.

        Returns:
            None: Returns object as NoneType

        """
        self.chat_model = ChatHuggingFace(llm=self.get_llm(), verbose=True)

    def get_chat_model(self) -> BaseChatModel:
        """
        Retrieves or constructs the chat model.

        Returns:
            BaseChatModel: The full chat model set up with the LLM.

        """
        if self.chat_model is None:
            self.set_chat_model()
        return self.chat_model
