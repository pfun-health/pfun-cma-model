from fastapi import FastAPI
from gradio import Interface, io
from pfun_cma_model.llm.llm import PFunLanguageModel, PromptContext

app = FastAPI()


@app.post("/pfun_cma_embed")
async def pfun_cma_embed(cgm_data: dict, context_name: str):
    agent = PFunLanguageModel(prompt_context=PromptContext.from_template("initial", context_name))
    agent.fit_model(cgm_data)
    responses = agent.get_llm_response()
    return responses


def gradio_ui():
    iface = Interface(
        fn=pfun_cma_embed,
        inputs=["text", "text"],
        outputs=[
            io.Textbox(label="Text Response"),
            io.Dataframe(label="Stats Table"),
            io.Dataframe(label="Schedule Info"),
            io.Plotly(label="Chart")
        ]
    )
    iface.launch()


if __name__ == "__main__":
    gradio_ui()
