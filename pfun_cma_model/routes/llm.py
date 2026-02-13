"""
PFun CMA Model - LLM API Routes
"""

import asyncio
import json

from fastapi import APIRouter, Response

from pfun_cma_model.llm import (
    generate_scenario as gen_scene
)
router = APIRouter()

DEFAULT_HEALTHY_PROMPT = """
This person is mostly healthy but occasionally eats a late dinner.
"""


@router.post("/generate-scenario")
async def generate_scenario(prompt: str = DEFAULT_HEALTHY_PROMPT, include_sample_trace: bool = False):
    """Use LLM endpoint to generate a realistic scenario (with hypothetical parameters).

    prompt: A natural language description of the scenario to generate (e.g. "a mostly healthy person who occasionally eats a late dinner").
    include_sample_trace: Whether to include a sample trace of blood glucose values for the generated scenario (this is optional since it can be expensive to generate).
    """

    async def attempt_scene_gen():
        response_data = await gen_scene(query=prompt, include_sample_trace=include_sample_trace)
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
    

