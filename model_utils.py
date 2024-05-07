import torch
from transformers import BitsAndBytesConfig
from transformers.utils.quantization_config import QuantizationConfigMixin


def get_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_use_double_quant: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
) -> QuantizationConfigMixin:
    """
    Configures and returns a quantization configuration object for use with transformer models, particularly using
    the BitsAndBytes method from Hugging Face transformers library.

    Args: load_in_4bit (bool):  Indicates whether to load the model in a 4-bit precision. Default is True.
    bnb_4bit_use_double_quant (bool): If True, enables double quantization during the quantization process. Default
    is True.
    bnb_4bit_quant_type (str): Specifies the type of quantization to use, typical options being "nf4" which stands
    for narrow 4-bit floating point. Default is "nf4".
    bnb_4bit_compute_dtype (torch.dtype):Determines the data type for computation during model inference. Typically,
    this can significantly affect performance and model size. Default is torch.bfloat16.

    Returns: QuantizationConfigMixin: A configuration mixin object that encapsulates the specified quantization
    parameters which can be applied to transformer models for efficient deployment.

    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )
    return bnb_config
