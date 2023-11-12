import os
from typing import (
    Optional
)
import pfun_path_helper as pph
pph.append_path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pfun_cma_model.config import settings
from pfun_cma_model.llm.context import PromptContext, InitialPromptContext
from pfun_cma_model.llm.llm import PFunLanguageModel
from pfun_cma_model.runtime.src.engine.fit import fit_model as fit_pfun_cma_model
from pfun_cma_model.runtime.src.engine.fit import CMAFitResult
from pfun_cma_model.llm.gen_summary import (
    generate_summary_content,
    ModelResult
)
import pandas as pd
import yaml


class PFunAgent:
    """LLM Agent for the PFun model.

    Associates a rendered PromptContext (with user data) to a PFunLanguageModel.
    """

    prompt_context: PromptContext = InitialPromptContext()
    
    def __init__(self,
                 template_name: str = "initial") -> None:
        self.prompt_context = PromptContext.from_template(template_name)
        self._pfun_model_result: Optional[CMAFitResult] = None
        self._user_data: Optional[pd.DataFrame] = None

    @property
    def llm(self):
        return PFunLanguageModel(self.prompt_context)

    def fit_pfun_cma_model(self, data: Optional[pd.DataFrame] = None):
        if data is None:
            data = self.load_user_data()
        return fit_pfun_cma_model(data)

    @property
    def pfun_model_result(self) -> CMAFitResult:
        if self._pfun_model_result is None:
            self._pfun_model_result = self.fit_pfun_cma_model(self._user_data)
        return self._pfun_model_result

    @classmethod
    def load_user_data(cls, user: Optional[dict] = None):
        """
        Load user data from a specified file path.

        :param user: A dictionary containing user data. If not provided, the user from the prompt context will be used.
        :type user: Optional[dict]
        :return: The loaded user data as a pandas DataFrame.
        :rtype: pd.DataFrame
        """
        if user is None:
            user = cls.prompt_context.user
        data_fpath = user['data_fpath']
        if not os.path.isabs(data_fpath):
            data_fpath = os.path.join(os.path.dirname(user['_user_fpath']), data_fpath)
        data = pd.read_csv(data_fpath)
        return data

    def generate_summary(self, result: Optional[CMAFitResult | ModelResult] = None):
        if result is None:
            result = self.pfun_model_result
        summary_dict = generate_summary_content(result, data=self._user_data)
        # convert dict to yaml string
        summary_str = yaml.dump(summary_dict, default_flow_style=False)
        return summary_str

    def __call__(self, data: Optional[pd.DataFrame] = None):
        """
        Call the agent with optional data. If data is provided, it will be used to fit the CMA model.

        Args:
            data (Optional[pd.DataFrame], optional): The input data. Defaults to None.

        Returns:
            self: The modified instance of the class.
        """
        if data is not None:
            self._user_data = data
        if self._user_data is None:
            self._user_data = self.load_user_data()
        #: fit the pfun cma model
        self._pfun_model_result = self.fit_pfun_cma_model(self._user_data)
        #: generate summary content
        summary_content = self.generate_summary(self._pfun_model_result)
        #: update user summary content in prompt context
        self.prompt_context.update_user_summary(summary_content)
        return self

    @property
    def text_response(self):
        text_response = self.llm.get_llm_response()
        return text_response


def main():
    agent = PFunAgent()
    agent()
    print(agent.text_response)
    return agent


if __name__ == "__main__":
    agent = main()