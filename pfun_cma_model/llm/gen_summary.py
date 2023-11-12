import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
import os
from fastapi.templating import Jinja2Templates
import pfun_path_helper
from pfun_path_helper import get_lib_path
from pfun_cma_model.runtime.src.engine.cma_posthoc import (
    RecsSummaryModel,
    ModelResult,
    calc_model_stats,
    ModelResultStats,
    CMIModel,
    CMIQuality,
    CMAPlotConfig
)
from pfun_cma_model.runtime.src.engine.fit import (
    CMAFitResult
)
import pandas as pd
from typing import Optional
from datetime import datetime

root_path = get_lib_path()

#: load templates
templates_path = os.path.join(root_path, "frontend", "templates")
templates = Jinja2Templates(directory=templates_path,
                            autoescape=False, auto_reload=True, enable_async=False)


def convert_to_model_result(model_result: CMAFitResult | ModelResult, data: Optional[pd.DataFrame] = None):
    """
    Convert the given `model_result` to an instance of the `ModelResult` class.

    Args:
        model_result (CMAFitResult | ModelResult): The result of the model fit.

    Returns:
        ModelResult: The converted model result.
    """
    if not isinstance(model_result, ModelResult):
        if data is None:
            raise ValueError("data must be provided if model_result is not an instance of ModelResult")
        values = {k: getattr(model_result, k) for k, _ in model_result.model_fields.items() if k in ModelResult.model_fields}
        if "formatted_data" not in values:
            values["formatted_data"] = data
        values.update({
            "userId": 'unknown',
            "queryDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": "success",
            "time": values['formatted_data']["time"].tolist(),
            "stats": ModelResultStats(**calc_model_stats(model_result.cma)),
            "popt": model_result.popt_named,
            "img_src": ''
        })
        model_result = ModelResult(**values)
    return model_result


def generate_summary_content(model_result: ModelResult | CMAFitResult, data: Optional[pd.DataFrame] = None):
    """
    Generates the summary content based on the provided model result.

    Args:
        model_result (ModelResult): The result of the model.

    Returns:
        dict: The generated summary content.
    """
    model_result = convert_to_model_result(model_result, data)
    summary_model = RecsSummaryModel(model_result=model_result)
    stats = dict(model_result.stats)
    stats["min_bg_value"] = model_result.formatted_data["value"].min()
    stats["max_bg_value"] = model_result.formatted_data["value"].max()

    summary_model = summary_model.model_dump()
    for category in ['weakness', 'strength']:
        category_obj = summary_model.get(category)
        if category_obj is None:
            logging.debug(f"Category {category} not found in summary model.")
            continue
        for text_part in ["fmt_short", "fmt_title"]:
            template_str = category_obj.get(text_part, "")
            rendered_out = templates.env.from_string(template_str).render(stats=stats)
            summary_model[category][text_part] = rendered_out

    content = dict(summary_model)
    content.pop("model_result")

    cmi: CMIModel = content.pop("cmi")
    content["cmi"] = dict(cmi)

    content["weaknesses"] = [clean_item(weakness) for weakness in content['weaknesses']]
    content["strengths"] = [clean_item(strength) for strength in content['strengths']]

    content = {
        "max_strength_comment": content["max_strength"]['rec_short'],
        "max_weakness_recommendation": content["max_weakness"]['rec_short'],
        "strengths": content["strengths"],
        "weaknesses": content["weaknesses"],
    }

    return content


def clean_item(item):
    item.pop("func")
    for key in list(item.keys()):
        if "func" in key:
            item.pop(key)
    return item
