from fastapi import FastAPI
from gradio import Interface
import gradio.components as io
import pfun_path_helper as pph
import os
pph.append_path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pfun_cma_model.config import settings
from pfun_cma_model.llm.agent import PFunAgent

app = FastAPI()




@app.post("/pfun_cma_embed")
async def pfun_cma_embed(template_name: str = 'initial', user_name: str = 'sample'):
    agent = PFunAgent(template_name)
    return [
        agent.text_response,
    ]


def gradio_ui():
    iface = Interface(
        fn=pfun_cma_embed,
        inputs=[
            io.Text(value='initial', placeholder='template_name'),
            io.Text(value='sample', placeholder='user_name')
        ],
        outputs=[
            io.Textbox(label="Text Response")
        ]
    )
    iface.launch(debug=True)


if __name__ == "__main__":
    gradio_ui()
