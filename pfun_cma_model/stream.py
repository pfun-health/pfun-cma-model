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


async def read_create_async_generator(fake_file) -> AsyncGenerator[str, None]:
    # Read lines in the fake_file asynchronously
    while True:
        line = fake_file.readline()
        if not line:
            break  # Exit when no more lines are available
        yield line.strip()  # Yield the line, removing any extra whitespace


async def stream_run_at_time_func(model: CMASleepWakeModel, t0: float | int, t1: float | int, n: int, **config) -> AsyncGenerator[str, None]:
    """calculate the glucose signal for the given timeframe and stream the results."""
    logger.debug(
        "(stream_run_at_time_func) Running model at time: t0=%s, t1=%s, n=%s, config=%s", t0, t1, n, config)
    bounded_params = {k: v for k,
                      v in config.items() if k in _BOUNDED_PARAM_KEYS_DEFAULTS}
    model.update(bounded_params)
    logger.debug(
        "(stream_run_at_time_func) Model parameters updated: %s", model.params)
    logger.debug(
        f"(stream_run_at_time_func) Generating time vector<{t0}, {t1}, {n}>...")
    t = model.new_tvector(t0, t1, n)
    df: DataFrame = model.calc_Gt(t=t)
    txt_buffer = StringIO()
    df.reset_index(inplace=True)
    df.rename(columns={"index": "t", }, inplace=True)
    df.to_csv(txt_buffer, header=False, index=False, columns=["t", "Gt"])
    txt_buffer.seek(0)
    async for t_Gt_pair in read_create_async_generator(txt_buffer):
        tx, Gty = t_Gt_pair.split(",")
        yield json.dumps({"x": tx, "y": Gty})