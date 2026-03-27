"""
Streaming-related functions for the CMA model.
"""

import json
import logging
from io import StringIO
from typing import AsyncGenerator
from pandas import DataFrame
from pfun_cma_model.engine.cma import CMASleepWakeModel
from pfun_cma_model.engine.cma_model_params import _BOUNDED_PARAM_KEYS_DEFAULTS

logger = logging.getLogger(__name__)


async def async_generate_parameters(
    description: str, llm_gen_scenario_endpoint: str
) -> str:
    """
    Asynchronously generate parameters for a scenario description.
    :param description: Description
    :param llm_gen_scenario_endpoint: Description
    """
    from httpx import AsyncClient

    logger.debug("Hitting llm generation endpoint: %s", str(llm_gen_scenario_endpoint))
    async with AsyncClient(timeout=30) as client:
        try:
            response = client.stream(
                method="POST",
                url=llm_gen_scenario_endpoint,
                json={"description": description},
                timeout=30,
            )
            if response.status_code == 200:
                # Successful response (JSON object)
                response_jdict = response.json()
                description_text = response_jdict.get("qualitative_description", "")
                import pandas as pd

                parameters = response_jdict.get("parameters", None)
                if parameters is not None:
                    param_df = pd.DataFrame.from_dict(parameters, orient="index")
                    param_df.index.name = "Parameter"
                    param_df.reset_index(inplace=True)
                    formatted_params_table = param_df.to_markdown(index=False)
                elif parameters is None:
                    formatted_params_table = "😞 No parameters generated.\n"
                    formatted_response = (
                        "## Description:\n"
                        f"{description_text}\n\n"
                        "## Generated Parameters:\n"
                        f"{formatted_params_table}\n"
                    )
                    return formatted_response
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Request failed: {e}"


async def read_create_async_generator(fake_file) -> AsyncGenerator[str, None]:
    # Read lines in the fake_file asynchronously
    while True:
        line = fake_file.readline()
        if not line:
            break  # Exit when no more lines are available
        yield line.strip()  # Yield the line, removing any extra whitespace


async def run_at_time_func(
    model: CMASleepWakeModel, t0: float | int, t1: float | int, n: int, **config
) -> str:
    """calculate the glucose signal for the given timeframe"""
    logger.debug(
        "(run_at_time_func) Running model at time: t0=%s, t1=%s, n=%s, config=%s",
        t0,
        t1,
        n,
        config,
    )
    bounded_params = {
        k: v for k, v in config.items() if k in _BOUNDED_PARAM_KEYS_DEFAULTS
    }
    model.update(bounded_params)
    logger.debug("(run_at_time_func) Model parameters updated: %s", model.params)
    logger.debug(f"(run_at_time_func) Generating time vector<{t0}, {t1}, {n}>...")
    t = model.new_tvector(t0, t1, n)
    df: DataFrame = model.calc_Gt(t=t)
    output = df.to_json()
    return output


async def stream_run_at_time_func(
    model: CMASleepWakeModel, t0: float | int, t1: float | int, n: int, **config
) -> AsyncGenerator[str, None]:
    """calculate the glucose signal for the given timeframe and stream the results."""
    logger.debug(
        "(stream_run_at_time_func) Running model at time: t0=%s, t1=%s, n=%s, config=%s",
        t0,
        t1,
        n,
        config,
    )
    bounded_params = {
        k: v for k, v in config.items() if k in _BOUNDED_PARAM_KEYS_DEFAULTS
    }
    model.update(bounded_params)
    logger.debug("(stream_run_at_time_func) Model parameters updated: %s", model.params)
    logger.debug(
        f"(stream_run_at_time_func) Generating time vector<{t0}, {t1}, {n}>..."
    )
    t = model.new_tvector(t0, t1, n)
    df: DataFrame = model.calc_Gt(t=t)
    txt_buffer = StringIO()
    df.reset_index(inplace=True)
    df.rename(
        columns={
            "index": "t",
        },
        inplace=True,
    )
    df.to_csv(txt_buffer, header=False, index=False, columns=["t", "Gt"])
    txt_buffer.seek(0)
    async for t_Gt_pair in read_create_async_generator(txt_buffer):
        tx, Gty = t_Gt_pair.split(",")
        yield json.dumps({"x": tx, "y": Gty}) + "\n"


async def stream_full_model_run(
    model: CMASleepWakeModel, t0: float | int, t1: float | int, n: int, **config
) -> AsyncGenerator[str, None]:
    """Calculate the full model run (c, m, a) for the given timeframe and stream the results."""
    logger.debug(
        "(stream_full_model_run) Running model at time: t0=%s, t1=%s, n=%s, config=%s",
        t0,
        t1,
        n,
        config,
    )
    bounded_params = {
        k: v for k, v in config.items() if k in _BOUNDED_PARAM_KEYS_DEFAULTS
    }
    model.update(bounded_params)
    logger.debug("(stream_full_model_run) Model parameters updated: %s", model.params)
    logger.debug(f"(stream_full_model_run) Generating time vector<{t0}, {t1}, {n}>...")
    t = model.new_tvector(t0, t1, n)

    # Update time vector on the model
    model.t = t

    # Create a DataFrame with the results we want
    df = DataFrame(
        {
            "t": model.t,
            "c": model.c,
            "m": model.m,
            "a": model.a,
        }
    )

    # We will stream this line by line like the other function
    txt_buffer = StringIO()
    df.to_csv(txt_buffer, header=False, index=False, columns=["t", "c", "m", "a"])
    txt_buffer.seek(0)
    async for line in read_create_async_generator(txt_buffer):
        parts = line.split(",")
        # Yield a JSON object with all values
        yield json.dumps(
            {"t": parts[0], "c": parts[1], "m": parts[2], "a": parts[3]}
        ) + "\n"
