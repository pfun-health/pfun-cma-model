from dataclasses import dataclass, field
from jinja2 import Template
from typing import Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from yaml import safe_load
import pfun_path_helper


class Jinja2Context:
    def __init__(self, template: str, user: Dict):
        self.template = Template(template)
        self.user = user

    def render(self) -> str:
        return self.template.render(**self.user)


@dataclass
class PromptContext(Jinja2Context):
    name: str = "initial"
    prompt_template: str = """Hi, {{ nickname }}! How are you feeling today?"""
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
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
        self.model = AutoModelForCausalLM.from_pretrained("stanford-crfm/BioMedLM")

    def create_embedding(self, prompt: str) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        return input_ids

    def generate_recommendations(self, prompt: Optional[str] = None,
                                 input_ids: Optional[str] = None) -> str:
        if all([x is None for x in [prompt, input_ids]]):
            raise ValueError("Both prompt and input_ids cannot be None")
        if input_ids is None:
            input_ids = self.create_embedding(prompt)
        output = self.model.generate(input_ids, max_length=150)
        recommendations = self.tokenizer.decode(
            output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        return recommendations
