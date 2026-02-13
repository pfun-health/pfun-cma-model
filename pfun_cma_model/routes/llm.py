"""
PFun CMA Model - LLM API Routes
"""

import asyncio
import json

from fastapi import APIRouter, Response

from pfun_cma_model.llm import generate_scenario as gen_scene

router = APIRouter()

DEFAULT_HEALTHY_PROMPT = """
This person is mostly healthy. They occasionally eat a late dinner (after 8pm) and sometimes skip breakfast. They have a moderate amount of stress in their life, but they manage it well. They get around 6-7 hours of sleep per night, but their sleep quality is not great. They do some light exercise a few times a week, but they are not very consistent with it.
"""


@router.post("/generate-scenario")
async def generate_scenario(
    prompt: str = DEFAULT_HEALTHY_PROMPT, include_sample_trace: bool = False, include_recommendations: bool = True
) -> Response:
    """Use LLM endpoint to generate a realistic scenario (with hypothetical parameters).

    prompt: A natural language description of the scenario to generate (e.g. "a mostly healthy person who occasionally eats a late dinner").
    include_sample_trace: Whether to include a sample trace of blood glucose values for the generated scenario (this is optional since it can be expensive to generate).
    include_recommendations: Whether to include recommendations in the generated scenario.
    """

    # sanitize the prompt (basic sanitation to prevent injection of malicious content into the LLM prompt - this is a simple example and should be improved for production use)
    prompt = prompt.strip()

    async def attempt_scene_gen():
        response_data = await gen_scene(
            query=prompt, include_sample_trace=include_sample_trace, include_recommendations=include_recommendations
        )
        try:
            content = json.dumps(response_data)
        except json.JSONDecodeError as exc:
            # try once more (after resting a moment)
            await asyncio.sleep(1)
            return await attempt_scene_gen()
        else:
            # if it works, return the content
            return content

    # perform the generation with a retry mechanism in case of JSON parsing errors (which can happen if the model's response is not well-formed JSON)
    content = await attempt_scene_gen()

    return Response(
        content=content,
        status_code=200,
        headers={"Content-Type": "application/json"},
    )
