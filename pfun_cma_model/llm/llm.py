import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional
from jinja2 import Template
from transformers import AutoModelForCausalLM, AutoTokenizer
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import json
import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding

import pfun_path_helper
from pfun_path_helper import get_lib_path
from pfun_cma_model.config import settings
from pfun_cma_model.runtime.src.engine.fit import fit_model as fit_pfun_cma_model


class Jinja2Context:
    def __init__(self, template: str = '', user: Optional[Dict] = None, summary_content: Optional[Dict] = None):
        self.template = Template(template)
        self.user = user
        self.summary_content = summary_content

    def render(self) -> str:
        return self.template.render(user=self.user, summary=self.summary_content)


@dataclass
class PFunUser:
    personal: Dict = field(default_factory=dict)
    has_dexcom: bool = True
    data_fpath: str = os.path.join(get_lib_path(), "..", "examples/data/valid_data.csv")

    def read_json(self, path: Optional[str] = None, inplace: bool = True):
        if path is None:
            path = os.path.join(get_lib_path(), "..", "examples/data/sample_user.json")
        data = {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if inplace is True:
            self.__dict__.update(data)
            return self
        return data

    @classmethod
    def dict(cls):
        if isinstance(cls, PFunUser):
            return dict(cls)
        inst = cls().read_json()
        return dict(inst)


@dataclass
class PromptContext(Jinja2Context):
    name: str = "dummy"
    template: str = """Hi, {{ user.nickname }}! How are you feeling today?"""
    user: Dict = field(default_factory=PFunUser.dict)
    summary_content: Dict = field(default_factory=dict)
    _prompts_dirpath: str = os.path.join(os.path.dirname(__file__), "prompts")

    def __post_init__(self):
        Jinja2Context.__init__(self, self.template, self.user, self.summary_content)

    @property
    def prompt(self):
        return self.render()

    @classmethod
    def from_template(cls, name: str, template: str = ""):
        return PromptContext(name=name, template=template)

    @classmethod
    def _check_args(cls, path: Optional[str] = None, name: Optional[str] = None):
        """
        Check the arguments passed to the yaml io functions.

        Args:
            path (Optional[str]): The path parameter (default: None).
            name (Optional[str]): The name parameter (default: None).

        Raises:
            RuntimeError: If neither the path nor the name is provided.

        Returns:
            Tuple[str, str]: The path and name after processing.
        """
        if all([path is None, name is None]):
            raise RuntimeError("Must provide a path or name!")
        if path is None:
            path = os.path.join(cls._prompts_dirpath, str(name))
        if '.yaml' != os.path.splitext(path)[1]:
            path = path + '.yaml'
        return path, name

    @classmethod
    def read_yaml(cls, path: Optional[str] = None, name: Optional[str] = None):
        """
        Read a YAML file and return a new instance of the class.

        Args:
            path (Optional[str]): The path to the YAML file. Defaults to None.
            name (Optional[str]): The name of the YAML file. Defaults to None.

        Returns:
            An instance of the class.

        Raises:
            FileNotFoundError: If the file is not found.
        """
        path, name = cls._check_args(path, name)
        config = {}
        with open(path, "r", encoding="utf-8") as f:
            config = load(f, Loader=Loader)
        return config

    def to_yaml(self, path: Optional[str] = None, name: Optional[str] = None):
        """
        Generate a YAML file from the object and save it to disk.

        Args:
            path (Optional[str]): The path where the YAML file will be saved. If not provided, the current directory will be used.
            name (Optional[str]): The name of the YAML file. If not provided, a default name will be used.

        Returns:
            str: The path of the saved YAML file.
        """
        path, name = self._check_args(path, name)  # type: ignore
        with open(path, "w", encoding="utf-8") as f:  # type: ignore
            dump(self.__dict__, f, Dumper=Dumper)
        return path


@dataclass
class InitialPromptContext(PromptContext):
    name: str = "initial"

    def __post_init__(self):
        super().__post_init__()
        config = self.read_yaml(name=self.name)
        self.template = config["template"]
        super().__post_init__()


class PFunLanguageModel:
    def __init__(self, prompt_context: PromptContext):
        self.tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained("stanford-crfm/BioMedLM")  # .to("cuda")
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

    def create_embedding(self, prompt: str) -> BatchEncoding:
        model_inputs = self.tokenizer(prompt, return_tensors="pt")  # .to("cuda")
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