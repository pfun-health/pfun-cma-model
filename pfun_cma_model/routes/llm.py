"""
PFun CMA Model - LLM API Routes
"""

from collections.abc import AsyncIterable, Iterable
import asyncio
import json
import pandas as pd
from fastapi import (
    APIRouter,
    Response,
    BackgroundTasks,
)
from fastapi.sse import (
    EventSourceResponse,
    ServerSentEvent,
)
from pydantic import BaseModel, ConfigDict, field_serializer
from pfun_common.settings import get_settings
from pfun_cma_model.llm import generate_scenario as gen_scene, GeneratedScenario
from pfun_cma_model.db import save2duckdb

router = APIRouter()

DEFAULT_HEALTHY_PROMPT = """
This person is mostly healthy. They occasionally eat a late dinner (after 8pm) and sometimes skip breakfast.
They have a moderate amount of stress in their life, but they manage it well.
They get around 6-7 hours of sleep per night, but their sleep quality is not great.
They do some light exercise a few times a week, but they are not very consistent with it.
"""


class SceneGenOptions(BaseModel):
    """Options for generating a scenario asynchronously"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str
    include_sample_trace: bool
    include_recommendations: bool
    background_tasks: BackgroundTasks

    @field_serializer("prompt")
    @classmethod
    def sanitize_prompt(cls, prompt):
        return prompt.strip()


async def attempt_scene_gen(options: SceneGenOptions) -> str:
    """
    Try to generate a scenario asynchronously, try again if the first attempt fails.
    """

    # sanitize the prompt (basic sanitation to prevent injection of malicious content into the LLM prompt - this is a simple example and should be improved for production use)
    prompt = options.prompt.strip()

    # Generate the scenario
    generated_scenario: GeneratedScenario = await gen_scene(
        query=prompt,
        include_sample_trace=options.include_sample_trace,
        include_recommendations=options.include_recommendations,
    )
    # convert to a JSON-seralizable dictionary
    response_data = generated_scenario.model_dump()

    # attempt to convert dict to JSON-serialized string
    try:
        content = json.dumps(response_data)
    except json.JSONDecodeError as exc:
        # try once more (after resting a moment)
        await asyncio.sleep(1)
        return await attempt_scene_gen(options)
    else:
        # if it succeeds, store the original dict in duckdb (background task)
        df_result = pd.DataFrame([response_data], index=[0])
        table_id = "cma_recs"
        db_path = (
            "results/duckdb-local.db"
            if get_settings().debug
            else "results/duckdb-remote.db"
        )
        options.background_tasks.add_task(
            save2duckdb, df_result=df_result, db_path=db_path, table_id=table_id
        )

    # return the content as a JSON serialized string
    return content


from typing import Sequence


@router.post("/generate-scenarios", response_class=EventSourceResponse)
async def generate_multiple_scenarios(
    background_tasks: BackgroundTasks,
    prompts: Sequence[str],
    include_sample_trace: bool = False,
    include_recommendations: bool = True,
) -> AsyncIterable[ServerSentEvent]:
    """Stream multiple scenarios asynchronously (streams SSEvents)."""
    for i, prompt in enumerate(prompts):
        options = SceneGenOptions(
            prompt=prompt,
            include_sample_trace=include_sample_trace,
            include_recommendations=include_recommendations,
            background_tasks=background_tasks,
        )
        scenario = attempt_scene_gen(options)
        yield ServerSentEvent(
            data=scenario, event="generated_scenario", id=str(i + 1), retry=2300
        )


@router.post("/generate-scenario")
async def generate_scenario(options: SceneGenOptions) -> Response:
    """Use LLM endpoint to generate a realistic scenario (with hypothetical parameters).

    prompt: A natural language description of the scenario to generate (e.g. "a mostly healthy person who occasionally eats a late dinner").
    include_sample_trace: Whether to include a sample trace of blood glucose values for the generated scenario (this is optional since it can be expensive to generate).
    include_recommendations: Whether to include recommendations in the generated scenario.
    """

    # Perform the generation with a retry mechanism in case of JSON parsing errors.
    # ...this can happen if the model's response is not well-formed JSON.
    content = await attempt_scene_gen(options)

    return Response(
        content=content,
        status_code=200,
        headers={"Content-Type": "application/json"},
    )
