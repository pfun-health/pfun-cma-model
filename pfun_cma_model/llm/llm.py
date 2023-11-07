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


def get_sample_user():
    """
    Retrieve a sample user from a JSON file.

    This function reads a JSON file containing a sample user and returns the user as a dictionary.

    Returns:
        dict: A dictionary representing the sample user.

    """
    user = {}
    with open(
        os.path.join(get_lib_path(), "..", "examples/data/sample_user.json"), "r", encoding="utf-8"
    ) as f:
        user = json.load(f)
    return user


@dataclass
class PromptContext(Jinja2Context):
    name: str = "dummy"
    prompt_template: str = """Hi, {{ user.nickname }}! How are you feeling today?"""
    user: Dict = field(default_factory=get_sample_user)
    summary_content: Dict = field(default_factory=dict)
    _prompts_dirpath: str = os.path.join(os.path.dirname(__file__), "prompts")

    def __post_init__(self):
        Jinja2Context.__init__(self, self.prompt_template, self.user, self.summary_content)

    @property
    def prompt(self):
        return self.render()

    @classmethod
    def from_template(cls, name: str, template: str = ""):
        return PromptContext(name=name, prompt_template=template)

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
        if path is None and name is not None:
            path = os.path.join(cls._prompts_dirpath, name)
        if '.yaml' != os.path.splitext(path)[1]:
            path = path + '.yaml'
        return path, name

    @classmethod
    def read_json(cls, path: Optional[str] = None, name: Optional[str] = None):
        path, name = cls._check_args(path, name)
        if path is None:
            raise RuntimeError("Must provide a path!")
        config = {}
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(**config)

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
        config = self.read_yaml(name=self.name)
        self.__dict__.update(config)


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

    def create_embedding(self, prompt: str) -> list[int]:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        return input_ids

    def generate_recommendations(
        self, prompt: Optional[str] = None, input_ids: Optional[list[int]] = None
    ) -> str:
        if prompt is None:
            prompt = self.prompt_context.prompt
        if input_ids is None:
            input_ids = self.create_embedding(prompt)
        if not isinstance(input_ids, np.ndarray):
            input_ids = np.array(input_ids)  # type: ignore
        input_ids = torch.from_numpy(input_ids).long().to('cuda')  # type: ignore
        output = self.model.generate(input_ids, max_length=150)
        recommendations = self.tokenizer.decode(
            output[:, input_ids.shape[-1] :][0], skip_special_tokens=True
        )
        return recommendations


if __name__ == "__main__":
    llm = PFunLanguageModel(InitialPromptContext())
    print(llm.get_llm_response())