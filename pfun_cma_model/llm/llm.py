from functools import lru_cache
from typing import Optional, Literal, Annotated

from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoModelForCausalLM, AutoTokenizer

import pfun_path_helper
from pfun_path_helper import get_lib_path
from pfun_cma_model.enums import StringEnum
from pfun_cma_model.config import settings
from pfun_cma_model.llm.context import PromptContext, InitialPromptContext


class PretrainedModel(StringEnum):
    """
    Pretrained LLMs from Hugging Face.
    """
    BIO_MED_LM = "stanford-crfm/BioMedLM"
    FB_BLENDERBOT = "facebook/blenderbot-400M-distill"
    DOLLY = "databricks/dolly-v2-7b"


class PFunLanguageModel:
    def __init__(self,
                 prompt_context: PromptContext,
                 pretrained_model: Annotated[str, PretrainedModel] = PretrainedModel.DOLLY,
                 device: Literal["cpu", "cuda"] = "cuda"):
        self._device = device
        self._pretrained_model = pretrained_model
        self.tokenizer = AutoTokenizer.from_pretrained(self._pretrained_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self._pretrained_model).to(self._device)
        self.prompt_context = prompt_context
        self.llm_response = None

    @lru_cache(maxsize=128)
    def get_llm_response(self, prompt_context: Optional[PromptContext] = None):
        if prompt_context is None:
            prompt_context = self.prompt_context
        if not self.llm_response or self.prompt_context != prompt_context:
            self.prompt_context = prompt_context
            self.llm_response = self.generate_recommendations()
        return self.llm_response

    def create_embedding(self, prompt: str) -> BatchEncoding:
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)
        return model_inputs

    def generate_recommendations(
        self, prompt: Optional[str] = None, model_inputs: Optional[BatchEncoding] = None
    ) -> str:
        if prompt is None:
            prompt = self.prompt_context.prompt
        if model_inputs is None:
            model_inputs = self.create_embedding(prompt)
        output = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=True,
                                     top_p=0.95, top_k=10, pad_token_id=self.tokenizer.pad_token_id)
        recommendations = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return recommendations


if __name__ == "__main__":
    context = InitialPromptContext()
    llm = PFunLanguageModel(context)
    print(llm.get_llm_response())