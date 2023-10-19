from fastapi import FastAPI
from gradio import Interface, io
from functools import lru_cache
import your_pfun_cma_model  # Assuming you have a module for your pfun-cma-model
import plotly.express as px
import pandas as pd

app = FastAPI()

class PFunCmaAgent:
    def __init__(self, cgm_data, demographic_info=None):
        self.embedding = your_pfun_cma_model.embed(cgm_data, demographic_info)
        self.llm_response = None

    @lru_cache(maxsize=128)
    def get_llm_response(self):
        if not self.llm_response:
            self.llm_response = your_pfun_cma_model.get_llm_response(self.embedding)
        return self.llm_response

    def get_responses(self):
        llm_response = self.get_llm_response()
        text_response = llm_response.get_text_response()

        stats_table = pd.DataFrame(llm_response.get_stats_table())
        schedule_info = pd.DataFrame(llm_response.get_schedule_info())

        cortisol_data = llm_response.get_cortisol_estimates()
        cortisol_chart = px.line(cortisol_data, x='time', y='cortisol_level')

        return text_response, stats_table, schedule_info, cortisol_chart

@app.post("/pfun_cma_embed")
async def pfun_cma_embed(cgm_data: dict, demographic_info: dict = None):
    agent = PFunCmaAgent(cgm_data, demographic_info)
    responses = agent.get_responses()
    return responses

def gradio_ui():
    iface = Interface(
        fn=pfun_cma_embed,
        inputs=["text", "text"],
        outputs=[
            io.Textbox(label="Text Response"),
            io.Dataframe(label="Stats Table"),
            io.Dataframe(label="Schedule Info"),
            io.Plotly(label="Cortisol Chart")
        ]
    )
    iface.launch()

if __name__ == "__main__":
    gradio_ui()
