from fastapi import FastAPI
from gradio import Interface
import gradio.components as io
from pfun_cma_model.llm.llm import PFunLanguageModel, PromptContext
from typing import Optional


app = FastAPI()


class PFunAgent:
    def __init__(self,
                 template_name: str,
                 raw_template: Optional[str] = None) -> None:
        self.prompt_context = PromptContext.from_template(template_name, raw_template)
        self.model = PFunLanguageModel(prompt_context=self.prompt_context)
        self._model_result = None

    @property
    def model_result(self):
        return self._model_result

    def __call__(self, data):
        self._model_result = self.model.fit_model(data)

    @property
    def text_response(self):
        text_response = self.model.get_llm_response()
        return text_response


@app.post("/pfun_cma_embed")
async def pfun_cma_embed(template_name: str = 'initial', raw_template: Optional[str] = None):
    agent = PFunAgent(template_name, raw_template=raw_template)
    return [
        agent.text_response,
    ]


def gradio_ui():
    iface = Interface(
        fn=pfun_cma_embed,
        inputs=[
            io.Text(value='initial', placeholder='template_name'),
            io.Timeseries()
        ],
        outputs=[
            io.Textbox(label="Text Response"),
            # io.Dataframe(label="Stats Table"),
            # io.Dataframe(label="Schedule Info")
        ]
    )
    iface.launch()


if __name__ == "__main__":
    gradio_ui()
