"""pfun_cma_model/llm.py: LLM prompting logic."""

import importlib
import json
import logging
import re
from typing import Optional, Any, Literal
from pfun_common.settings import get_settings
from pfun_cma_model.engine.cma_model_params import CMAModelParams


LLMBackendChoice = Literal[
    "google", "perplexity", "ollama", "openai"
]  # The allowed choices for LLM backend, corresponding to the implemented backends in pfun_llm.backend.


def _import_genai_with_backend(llm_backend: LLMBackendChoice):
    """Dynamically import the currently selected LLM backend (using settings.llm_backend)."""
    module_name = f"pfun_llm.backend.{llm_backend}"
    class_name = f"{llm_backend}".title() + "GenerativeModel"
    _module = importlib.import_module(module_name)
    return getattr(_module, class_name)


def init_gen_model(**kwds):
    """Initializes the generative model based on the selected backend and provided keyword arguments.

    :param kwds: Keyword arguments to pass to the generative model upon initialization (e.g. temperature, seed, etc.). These will be passed directly to the model's internal _extra_kwds dictionary, which is used to configure the model's behavior.
    """
    kwargs = dict(options={"temperature": 0, "seed": 23})
    kwargs.update(kwds)
    GenerativeModel = _import_genai_with_backend(get_settings().llm_backend)
    model = GenerativeModel()
    model._extra_kwds.update(kwargs)
    return model


GenerativeModel = init_gen_model


async def _parse_generated_response(response: Any | str) -> str:
    """Parse the response that was returned by the generative model.
    Await the future if it's an async routine-like object.
    Get the response text attribute if it exists, otherwise return the string.
    """
    # explicitly test to see if the response needs awaited
    if not hasattr(response, "__await__"):
        # parse text attribute if it exists
        txt_resp = getattr(response, "text", str(response))
        # Properly handle UTF-8 encoding: encode to bytes then decode as UTF-8
        txt_resp = str(txt_resp).replace("'", '"')
        # strip surrounding formatting if it's wrapped in markdown code blocks
        txt_resp = re.sub(r"^```[\w\s]*|```$", "", txt_resp.strip())
        try:
            # If it's a string with encoding issues, try to fix it
            if isinstance(txt_resp, str):
                # Encode as latin-1 (single bytes) then decode as UTF-8 to fix double-encoding
                txt_resp = txt_resp.encode("utf-8", errors="replace").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            # If that fails, just use the string as-is
            pass
        return txt_resp
    return await _parse_generated_response(await response)


async def _call_llm_for_json(prompt: str) -> dict:
    """
    Calls the generative model with a prompt and parses the JSON response.

    Args:
        prompt: The prompt to send to the model.

    Returns:
        A dictionary parsed from the model's JSON response.

    Raises:
        Exception: If the API response cannot be parsed as JSON.
    """
    model = GenerativeModel()
    response = model.generate_content(prompt)
    resp_text: str = await _parse_generated_response(response)
    logging.debug("LLM Response (raw text attribute):\n'%s'", resp_text)
    try:
        # attempt to load without parsing
        resp_dict = json.loads(resp_text)
        resp_text = resp_dict["content"]
    except (json.JSONDecodeError, KeyError) as e:
        logging.debug("Failed in initial pre-parsing, attempting without...", exc_info=True)

    # use regex to extract JSON from markdown code blocks (if present)
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", resp_text, re.DOTALL)
    try:
        # The response might contain markdown, so we need to extract the JSON from it
        json_str = json_match.group(1) if json_match else resp_text.strip().replace("`", "").replace("json", "")
        # Ensure proper UTF-8 handling without aggressive escaping
        return json.loads(json_str)
    except (json.JSONDecodeError, KeyError, AttributeError, IndexError) as e:
        logging.debug("Raw response text:\n%s", resp_text)
        logging.error("Failed to parse LLM API Response. %s", e, exc_info=True)
        raise Exception(f"Failed to parse LLM API response: {e}")


async def generate_scenario(
    query: Optional[str] = None, include_sample_trace: bool = False, include_recommendations: bool = True
) -> dict:
    """
    Generates a realistic "pfun-scene" JSON object using the Gemini API.

    Args:
        query: An optional query to guide the scenario generation.
        include_sample_trace: Whether to include a sample trace in the generated scenario.
        include_recommendations: Whether to include recommendations in the generated scenario.

    Returns:
        A dictionary containing the generated scenario.
    """

    # Construct the prompt

    # baseline cma model parameters
    basal_params = CMAModelParams()
    basal_param_descriptions = basal_params.generate_markdown_table(output_fmt="md")

    # hypothetical scenario-conditioned parameters
    scenario_params = CMAModelParams(Cm=1.5, B=0.001, taug=0.4, tM=[7, 11, 18])
    scenario_description = (
        "This individual is experiencing a period of high stress due to work deadlines, "
        "which has been disrupting their sleep patterns and leading to poor dietary choices, especially in the evenings."
        " \nTheir cortisol levels are elevated, which is causing increased glucose variability;"
        " Their cortisol sensitivity (Cm={}) is {} from the expected {}."
        " \nTheir diet lacks high-quality proteins & fats, so their endogenous glucose production is dangerously unreliable. "
        "Thus, their baseline glucose (B={}) is {} compared to the expected {}; "
        "Further, their rate of glucose metabolism (taug={}) is {} compared to the expected {}. "
        " \nCombined with the physiological effects of stress, they have an increased risk of experiencing episodes of nocturnal hypoglycemia, i.e. dangerously low blood glucose levels."
    ).format(
        scenario_params.Cm,
        scenario_params.generate_qualitative_descriptor("Cm"),
        basal_params.Cm,
        scenario_params.B,
        scenario_params.generate_qualitative_descriptor("B"),
        basal_params.B,
        scenario_params.taug,
        scenario_params.generate_qualitative_descriptor("taug"),
        basal_params.taug,
    )

    scenario_param_descriptions = scenario_params.generate_markdown_table(
        output_fmt="md",
        included_params=["Cm", "B", "taug"],
        # Only include the parameters that are relevant,
        # exclude tM because it isn't a scalar bounded parameter.
    )

    # construct the prompt with recommendations (if included)
    include_tips_prompt = (
        "Ensure the recommendations include actionable tips to help the person mitigate their risk of hypoglycemia, such as stress management techniques, dietary adjustments, or sleep hygiene improvements. Important: the generated recommendations should be physiologically sound and appropriate for the scenario, and should not include generic advice that isn't relevant to the specific scenario; in most cases, the recommendations should map cleanly to specific parameter deviations and the qualitative description of the scenario."
        if include_recommendations
        else ""
    )
    recommendations_json_extra = (
        f',\n    "recommendations": "A concise list of personalized recommendations for the person based on the generated scenario. {include_tips_prompt}"'
        if include_recommendations
        else ""
    )
    specific_recommendations_json_extra = ""
    if include_recommendations:
        specific_recommendations_json_extra = f"""\
            ,
            "recommendations": {{
                "stress_management": "Employ deep-breathing exercises to manage stress.",
                "dietary_adjustments": "Include high-quality proteins and fats in evening meals to stabilize glucose levels, thus avoiding hypoglycemic episodes. Positive clinical outcomes should result in significantly decreased Cm, ideally closer to the expected baseline ({basal_params.Cm:.2f}).",
                "sleep_hygiene_improvements": "Aim to maintain a consistent sleep schedule and avoid screens before bedtime. Improved sleep quality can help regulate cortisol levels, thus decreasing overall glucose variability (positive outcomes are seen in a return to baseline Cm). Aim to get at least 7 hours of sleep per night; increased sleep duration can also help stabilize the global rate of postprandial glucose metabolism (taug, baseline expected value {basal_params.taug:.2f}); this helps mitigate hypoglycemia risk by increasing the time until glucose levels return to baseline (or drop dangerously low)."
        }}"""

    prompt = f"""\
You are a helpful assistant that generates realistic scenarios for a person with diabetes.
The user will provide a query to guide the generation.
If the query appears blank, then generate a realistic hypothetical scenario.
All generations must be completely valid physiofunctional results.

You will return a JSON object with the following structure:
```json
{{
    "forecasted_events": "A concise list of predicted health events.",
    "qualitative_description": "A concise clinical description of the person's metabolic health, lifestyle, and any recent health-relevant events.",
    "parameters": {{
        "param1": {{
            "value": value1, "description": "Description of param1"
        }},
        "param2": {{
            "value": value2, "description": "Description of param2"
        }},
        ...
    }}{recommendations_json_extra}
}}
```
Here are the baseline PFun CMA model parameters, displayed as a markdown-formatted table:
{basal_param_descriptions}

Now consider a case when the user requests a non-baseline scenario-conditioned PFun CMA model parameters:
User: "a patient with chronic stress that exacerbates the risk of glucose lows in the evening"
Think: "Corresponding to the scenario, here is a hypothetical scenario-conditioned PFun CMA model parameters: "
{scenario_param_descriptions}
Assistant:
```json
{{
    "forecasted_events": "Low blood glucose (hypoglycemic episodes) in the evening",
    "qualitative_description": "{scenario_description}",
    "parameters": {{
        "Cm": {{ "value": {scenario_params.Cm},  "stderr": {scenario_params.serr("Cm")}, "description": "Heightened stress level, leading to increased cortisol-mediated glucose variability" }},
        "B": {{ "value": {scenario_params.B}, "stderr": {scenario_params.serr("B")}, "description": "Low baseline glucose" }},
        "tM": {{ "value": [7, 11, 18], "description": "Consistent meal times throughout the day, keep up the great work! Consider eating a small snack after dinner to avoid hypoglycemia at night." }}
    }}{specific_recommendations_json_extra}
}}
```

Now, please generate a scenario based on the following user query. If the query is empty, generate a random scenario.
User: "{query if query else 'No query provided.'}"
Assistant:
"""
    return await _call_llm_for_json(prompt)
