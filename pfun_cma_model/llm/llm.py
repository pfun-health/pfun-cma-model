from dataclasses import dataclass, field
from functools import lru_cache
from jinja2 import Template
from typing import Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from yaml import safe_load
import pfun_path_helper
from pfun_cma_model.runtime.src.engine.fit import fit_model as fit_pfun_cma_model



class Jinja2Context:
    def __init__(self, template: str, user: Dict, summary_content: Dict):
        self.template = Template(template)
        self.user = user
        self.summary_content = summary_content

    def render(self) -> str:
        return self.template.render(user=self.user, summary=self.summary_content)


@dataclass
class PromptContext(Jinja2Context):
    name: str = "initial"
    prompt_template: str = """Hi, {{ user.nickname }}! How are you feeling today?"""
    user: Dict = field(default_factory=dict)
    summary_content: Dict = field(default_factory=dict)

    @property
    def prompt(self):
        return self.render()

    @classmethod
    def from_template(cls, name: str, template: str):
        return PromptContext(name=name, prompt_template=template)

    @classmethod
    def read_yaml(cls, path: str):
        config = {}
        with open(path, "r", encoding="utf-8") as f:
            config = safe_load(f)
        return cls(**config)


class PFunLanguageModel:
    def __init__(self, prompt_context: PromptContext):
        self.tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
        self.model = AutoModelForCausalLM.from_pretrained("stanford-crfm/BioMedLM")
        self.prompt_context = prompt_context
        self.llm_response = None
        self._pfun_model = None

    def fit_model(self, data):
        self._pfun_model = fit_pfun_cma_model(data)
        return self._pfun_model

    @lru_cache(maxsize=128)
    def get_llm_response(self, prompt_context: Optional[PromptContext] = None):
        if prompt_context is None:
            prompt_context = self.prompt_context
        if not self.llm_response or self.prompt_context != prompt_context:
            self.prompt_context = prompt_context
            self.llm_response = self.generate_recommendations()
        return self.llm_response

    def create_embedding(self, prompt: str) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        return input_ids

    def generate_recommendations(self, prompt: Optional[str] = None,
                                 input_ids: Optional[str] = None) -> str:
        if prompt is None:
            prompt = self.prompt_context.prompt
        if input_ids is None:
            input_ids = self.create_embedding(prompt)
        output = self.model.generate(input_ids, max_length=150)
        recommendations = self.tokenizer.decode(
            output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        return recommendations
