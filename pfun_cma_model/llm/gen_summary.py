import os
from fastapi.templating import Jinja2Templates
import pfun_path_helper
from pfun_path_helper import get_lib_path
from pfun_cma_model.runtime.src.engine.cma_posthoc import RecsSummaryModel
root_path = get_lib_path()

#: load templates
templates_path = os.path.join(root_path, "frontend", "templates")
templates = Jinja2Templates(directory=templates_path,
                            autoescape=False, auto_reload=True, enable_async=False)


def gen_summary_content(model_result):
    recs_summary = RecsSummaryModel(model_result=model_result)
    stats = model_result.stats.dict()
    stats["min_bg_value"] = model_result.formatted_data["value"].min()
    stats["max_bg_value"] = model_result.formatted_data["value"].max()
    for strweak in ['weakness', 'strength']:
        strweak_obj = recs_summary.get(strweak)
        for textpart in ["fmt_short", "fmt_title"]:
            template_str = strweak_obj.get(textpart, "")
            rendered_out = templates.env.from_string(
                template_str).render(stats=stats)
            recs_summary[strweak][textpart] = rendered_out
    content = dict(recs_summary)
    content.pop("model_result")
    content.pop("cmi")
    wsout = []
    weaknesses = content['weaknesses']
    for weakness in weaknesses:
        weakness.pop("func")
        for key in list(weakness.keys()):
            if "func" in key:
                weakness.pop(key)
        wsout.append(weakness)
    content["weaknesses"] = wsout
    ssout = []
    strengths = content['strengths']
    for strength in strengths:
        strength.pop("func")
        for key in list(strength.keys()):
            if "func" in key:
                strength.pop(key)
        ssout.append(strength)
    content["strengths"] = ssout
    return content
