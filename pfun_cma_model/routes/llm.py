"""
PFun CMA Model - LLM API Routes
"""
from fastapi import APIRouter, Response
from pfun_cma_model.llm import (
    generate_scenario as gen_scene,
    translate_query_to_params as translate_query
)
import json

router = APIRouter()

DEFAULT_HEALTHY_PROMPT = """
This person is mostly healthy but occasionally eats a late dinner.
"""

@router.post("/generate-scenario")
def generate_scenario(prompt: str = DEFAULT_HEALTHY_PROMPT):
    """Use VertexAI LLM endpoint to generate a realistic scenario (with hypothetical parameters)."""
    response_data = gen_scene(query=prompt)
    return Response(
        content=json.dumps(response_data),
        status_code=200,
        headers={"Content-Type": "application/json"},
    )


@router.post("/translate-query")
def translate_query_to_params(prompt: str = DEFAULT_HEALTHY_PROMPT):
    """Use gemini to translate the given scenario to a set of pfun-cma-model parameters."""
    response_data = translate_query(query=prompt)
    return Response(
        content=json.dumps(response_data),
        status_code=200,
        headers={"Content-Type": "application/json"},
    )