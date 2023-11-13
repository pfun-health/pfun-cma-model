import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
import os
from fastapi.templating import Jinja2Templates
import pfun_path_helper
from pfun_path_helper import get_lib_path
from pfun_cma_model.runtime.src.engine.cma_posthoc import (
    calc_model_stats
)
from pfun_cma_model.runtime.src.engine.fit import (
    CMAFitResult
)
import pandas as pd
from typing import Optional
from datetime import datetime

root_path = get_lib_path()




def generate_summary_content(model_result: CMAFitResult, data: Optional[pd.DataFrame] = None):
    """
    Generates the summary content based on the provided model result.

    Args:
        model_result (ModelResult): The result of the model.

    Returns:
        dict: The generated summary content.
    """
    content = {}

    return content


def clean_item(item):
    item.pop("func")
    for key in list(item.keys()):
        if "func" in key:
            item.pop(key)
    return item
