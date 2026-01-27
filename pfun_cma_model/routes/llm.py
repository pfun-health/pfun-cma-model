"""
PFun CMA Model - LLM API Routes
"""
import asyncio
import json

from fastapi import APIRouter, Response

from pfun_cma_model.llm import generate_scenario as gen_scene
from pfun_cma_model.llm import translate_query_to_params as translate_query

router = APIRouter()

DEFAULT_HEALTHY_PROMPT = """
This person is mostly healthy but occasionally eats a late dinner.
"""


@router.post("/generate-scenario")
async def generate_scenario(prompt: str = DEFAULT_HEALTHY_PROMPT):
    """Use LLM endpoint to generate a realistic scenario (with hypothetical parameters)."""

    async def attempt_scene_gen():
        response_data = await gen_scene(query=prompt)
        try:
            content = json.dumps(response_data)
        except json.JSONDecodeError as exc:
            # try once more (after resting a moment)
            await asyncio.sleep(1)
            return await attempt_scene_gen()
        else:
            # if it works, return the content
            return content
    
    content = await attempt_scene_gen()
    
    return Response(
        content=content,
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
