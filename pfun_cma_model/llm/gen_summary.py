import os
from fastapi.templating import Jinja2Templates
import pfun_path_helper
from pfun_path_helper import get_lib_path
from pfun_cma_model.runtime.src.engine.cma_posthoc import (
    RecsSummaryModel,
    ModelResult,
    CMIModel,
    CMIQuality,
    CMAPlotConfig
)

root_path = get_lib_path()

#: load templates
templates_path = os.path.join(root_path, "frontend", "templates")
templates = Jinja2Templates(directory=templates_path,
                            autoescape=False, auto_reload=True, enable_async=False)


def generate_summary_content(model_result: ModelResult):
    """
    Generates the summary content based on the provided model result.

    Args:
        model_result (ModelResult): The result of the model.

    Returns:
        dict: The generated summary content.
    """
    summary_model = RecsSummaryModel(model_result=model_result)
    stats = model_result.stats.model_dump()
    stats["min_bg_value"] = model_result.formatted_data["value"].min()
    stats["max_bg_value"] = model_result.formatted_data["value"].max()

    for category in ['weakness', 'strength']:
        category_obj = summary_model.get(category)
        for text_part in ["fmt_short", "fmt_title"]:
            template_str = category_obj.get(text_part, "")
            rendered_out = templates.env.from_string(template_str).render(stats=stats)
            summary_model[category][text_part] = rendered_out

    content = dict(summary_model)
    content.pop("model_result")

    cmi: CMIModel = content.pop("cmi")
    content["cmi"] = cmi.model_dump()

    content["weaknesses"] = [clean_item(weakness) for weakness in content['weaknesses']]
    content["strengths"] = [clean_item(strength) for strength in content['strengths']]

    return content


def clean_item(item):
    item.pop("func")
    for key in list(item.keys()):
        if "func" in key:
            item.pop(key)
    return item
