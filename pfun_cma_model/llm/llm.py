from transformers import AutoTokenizer, AutoModelForCausalLM
from pfun_cma_model.runtime.chalicelib.engine.cma_posthoc import RecsSummaryModel
from fastapi.templating import Jinja2Templates
import os

#: load templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"),
                            autoescape=False, auto_reload=True, enable_async=False)


def create_prompt(model_result):
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
    if bool(int(raw)) is False:
        context = dict(request=request, recs=None,
                       summary=recs_summary, stats=stats)
        return templates.TemplateResponse("recs_summary.html.jinja2", context)
    else:
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
    return prompt

class PFunLanguageModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
        self.model = AutoModelForCausalLM.from_pretrained("stanford-crfm/BioMedLM")

    def create_embedding(self, prompt: str) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        return input_ids

    def generate_recommendations(self, prompt: str) -> str:
        input_ids = self.create_embedding(prompt)
        output = self.model.generate(input_ids, max_length=150)
        recommendations = self.tokenizer.decode(
            output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        return recommendations
    