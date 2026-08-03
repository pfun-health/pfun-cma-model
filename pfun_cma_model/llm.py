"""pfun_cma_model/llm.py: LLM prompting logic."""

import importlib
import json
import logging
import re
from typing import Optional, Any, Literal, List
from pydantic import BaseModel, PrivateAttr
from pfun_common.settings import get_settings
from pfun_cma_model.engine.cma_model_params import CMAModelParams

logger = logging.getLogger(__name__)

LLMBackendChoice = Literal[
    "google", "perplexity", "ollama", "openai"
]  # The allowed choices for LLM backend, corresponding to the implemented backends in pfun_llm.backend.


class DescribedParameter(BaseModel):
    """
    PFun model parameter, along with descriptioon, value, standard error estimate.
    """

    value: float | int | Any
    #: Parameter value.

    description: str
    #: Text description.

    stderr: float
    #: Standard error estimate.


QualitativeLiteral = Literal["normal ", "low", "high", "very low", "very high"]
#: Literals for qualitative string


class PFunHealthInfo(BaseModel):
    """
    Defines the expected schema for estimated user health info.
    """

    age: int
    #: Estimated age of the individual

    sex: Literal["f", "m", "o"]
    #: Estimated biological sex of the individual

    stress_level: QualitativeLiteral
    #: Estimated stress level of the individual

    sleep_quality: QualitativeLiteral
    #: Estimated sleep quality of the individual

    circadian_misalignment: QualitativeLiteral
    #: Estimated circadian misalignment of the individual


class PFunLLMGeneratedScenario(BaseModel):
    """
    Defines the expected schema for an LLM-Generated scenario.
    """

    forecasted_events: str
    #: A concise list of predicted health events.

    qualitative_description: str
    #: A concise clinical description of the person's metabolic health, lifestyle, and any recent health-relevant events.

    parameters: dict[str, DescribedParameter]
    #: A mapping of pfun model parameter names with corresponding value, description, and stderr.

    health_info: PFunHealthInfo
    #: Health information for the given individual.

    recommendations: dict[str, str]
    #: A mapping of pfun llm generated recommendations, indexed by recommendation-type.

    _used_fallback_health_info: bool = PrivateAttr(default=False)
    #: True when ``health_info`` was not produced by the LLM but substituted with the fixed
    #: sample profile (see :func:`generate_scenario`). Computed by ``generate_scenario``; the
    #: LLM never emits this value. As a PrivateAttr it is automatically excluded from
    #: ``model_dump()`` and the JSON schema, so downstream persistence (e.g. duckdb) is
    #: unaffected.

    @property
    def used_fallback_health_info(self) -> bool:
        """Whether the fixed sample health profile was substituted for the LLM response."""
        return self._used_fallback_health_info

    @used_fallback_health_info.setter
    def used_fallback_health_info(self, value: bool) -> None:
        self._used_fallback_health_info = value


GeneratedScenario = PFunLLMGeneratedScenario
#: Alias for PFunLLMGeneratedScenario


def _import_genai_with_backend(llm_backend: LLMBackendChoice):
    """Dynamically import the currently selected LLM backend (using settings.llm_backend)."""
    module_name = f"pfun_llm.backend.{llm_backend}"
    class_name = f"{llm_backend}".title() + "GenerativeModel"
    _module = importlib.import_module(module_name)
    return getattr(_module, class_name)


def init_gen_model(**kwds):
    """Initializes the generative model based on the selected backend and provided keyword arguments.

    :param kwds: Keyword arguments to pass to the generative model upon initialization (e.g. temperature, seed, etc.).
                 These will be passed directly to the model's internal _extra_kwds dictionary,
                 which is used to configure the model's behavior.
    """
    kwargs = {
        "options": {"temperature": 0, "seed": 23},
        "format": PFunLLMGeneratedScenario.model_json_schema(),  # specifies the expected output format exactly
    }
    kwargs.update(kwds)
    GenerativeModel = _import_genai_with_backend(get_settings().llm_backend)
    model = GenerativeModel()
    model._extra_kwds.update(kwargs)
    return model


GenerativeModel = init_gen_model
#: alias (to clearly indicate this results in a new class instance)


def _to_described_parameters(params: CMAModelParams) -> dict[str, DescribedParameter]:
    """Build described parameter objects from CMA model parameters."""
    bounded_descriptions = dict(
        zip(params.bounded_param_keys, params.bounded_param_descriptions)
    )
    described_parameters: dict[str, DescribedParameter] = {}
    for name, value in params.model_dump().items():
        described_parameters[name] = DescribedParameter(
            value=value,
            description=bounded_descriptions.get(name, f"{name} parameter"),
            stderr=float(params.calc_serr(name)) if name in params.bounded_param_keys else 0.0,
        )
    return described_parameters


async def _parse_generated_response(response: Any | str) -> str:  # type: ignore
    """Parse the response that was returned by the generative model.
    Await the future if it's an async routine-like object.
    Get the response text attribute if it exists, otherwise return the string.
    """
    # explicitly test to see if the response needs awaited
    if not hasattr(response, "__await__"):
        # parse text attribute if it exists
        response_as_dict = response.model_dump()  # type: ignore
        txt_resp = response_as_dict["message"]["content"]
        # Properly handle UTF-8 encoding: encode to bytes then decode as UTF-8
        txt_resp = str(txt_resp).replace("'", '"')
        try:
            # If it's a string with encoding issues, try to fix it
            if isinstance(txt_resp, str):
                # Encode, then decode UTF-8 to fix double-encoding
                txt_resp = txt_resp.encode("utf-8", errors="replace").decode(
                    "utf-8", errors="replace"
                )
        except (UnicodeDecodeError, UnicodeEncodeError):
            # If that fails, just use the string as-is
            logging.debug(
                "Exception occurred during unicode handling, using original text response",
                exc_info=True,
            )
            logging.debug(
                "Failed to properly decode response text, using raw string.\nOriginal text: %s",
                txt_resp,
            )
        return txt_resp
    elif hasattr(response, "__await__"):
        return await _parse_generated_response(await response)  # type: ignore


async def _call_llm_for_json(prompt: str, stream: bool = False) -> dict:
    """
    Calls the generative model with a prompt and parses the JSON response.

    Args:
        prompt: The prompt to send to the model.

    Kwargs:
        stream [bool] : flag to indicate whether to stream the chat interaction or not.

    Returns:
        A dictionary parsed from the model's JSON response.

    Raises:
        Exception: If the API response cannot be parsed as JSON.
    """
    model = GenerativeModel()
    response = model.generate_content(prompt, stream=stream)
    resp_text: str = await _parse_generated_response(response)
    logging.debug("LLM Response (raw text attribute):\n'%s'", resp_text)
    # use regex to extract JSON from markdown code blocks (if present)
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", resp_text, re.DOTALL)
    # Perform additional cleaning to handle cases where the model might return JSON without proper code blocks,
    # or with extra text. This is a fallback in case the regex doesn't find a code block,
    # or if the model returns something like "Here is the JSON: { ... }".
    json_str = (
        json_match.group(1)
        if json_match
        else resp_text.strip().replace("`", "").replace("json", "")
    )
    json_str = json_str.strip()
    json_end_idx = json_str.rfind("}")
    if json_end_idx != -1:
        json_str = json_str[: json_end_idx + 1]
    logging.debug(
        "Extracted JSON string from response:\n'%s'",
        json_match.group(1) if json_match else "No JSON code block found.",
    )
    try:
        # Load the JSON string into a Python dictionary
        return json.loads(json_str)
    except (json.JSONDecodeError, KeyError, AttributeError, IndexError) as e:
        logging.debug("Raw response text:\n%s", resp_text)
        logging.error("Failed to parse LLM API Response. %s", e, exc_info=True)
        raise Exception(f"Failed to parse LLM API response: {e}")


async def generate_scenario(
    query: Optional[str] = None,
    include_sample_trace: bool = False,
    include_recommendations: bool = True,
    stream: bool = False,
) -> PFunLLMGeneratedScenario:
    """
    Generates a realistic "pfun-scene" JSON object using the selected llm backend (see pfun_common.settings).

    Args:
        query: An optional query to guide the scenario generation.
        include_sample_trace: Whether to include a sample trace in the generated scenario.
        include_recommendations: Whether to include recommendations in the generated scenario.

    Returns:
        A PFunLLMGeneratedScenario containing the generated scenario. The
        ``used_fallback_health_info`` attribute is True when the LLM omitted
        ``health_info`` and a fixed sample profile was substituted instead; it
        is False whenever ``health_info`` was produced by the LLM.
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
        "Generate a JSON object whose keys correspond to recommendation types (e.g., stress_reduction, dietary), and whose values contain corresponding specific recommendations. Ensure the recommendations include actionable tips to help the person mitigate their risk of hypoglycemia, such as stress management techniques, dietary adjustments, or sleep hygiene improvements. Important: the generated recommendations should be physiologically sound and appropriate for the scenario, and should not include generic advice that isn't relevant to the specific scenario; in most cases, the recommendations should map cleanly to specific parameter deviations and the qualitative description of the scenario. Extremely important: ensure the recommendations are returned as a JSON dictionary object, NOT a bulleted or numbered list!"
        if include_recommendations
        else ""
    )
    recommendations_json_extra = (
        f',\n    "recommendations": "A concise list of personalized recommendations for the person based on the generated scenario. {include_tips_prompt}"'
        if include_recommendations
        else ""
    )

    sample_recommendations = (
        {
            "stress_management": "Employ deep-breathing exercises to manage stress.",
            "dietary_adjustments": "Include high-quality proteins and fats in evening meals to stabilize glucose levels, thus avoiding hypoglycemic episodes. Positive clinical outcomes should result in significantly decreased Cm, ideally closer to the expected baseline ({basal_params.Cm:.2f}).",
            "sleep_hygiene_improvements": "Aim to maintain a consistent sleep schedule and avoid screens before bedtime. Improved sleep quality can help regulate cortisol levels, thus decreasing overall glucose variability (positive outcomes are seen in a return to baseline Cm). Aim to get at least 7 hours of sleep per night; increased sleep duration can also help stabilize the global rate of postprandial glucose metabolism (taug, baseline expected value {basal_params.taug:.2f}); this helps mitigate hypoglycemia risk by increasing the time until glucose levels return to baseline (or drop dangerously low).",
        }
        if include_recommendations
        else {}
    )

    sample_health_info = PFunHealthInfo(
        age=37,
        sex="f",
        stress_level="high",
        sleep_quality="low",
        circadian_misalignment="high",
    )
    sample_generated_scenario = PFunLLMGeneratedScenario(
        forecasted_events="Low blood glucose (hypoglycemic episodes) in the evening",
        qualitative_description=str(scenario_description),
        parameters=_to_described_parameters(scenario_params),
        health_info=sample_health_info.model_dump(),
        recommendations=sample_recommendations,
    )

    prompt = f"""\
You are a helpful assistant that generates realistic scenarios for a person with diabetes.
The user will provide a query to guide the generation.
If the query appears blank, then generate a realistic hypothetical scenario.
All generations must be completely valid physiofunctional results.

You will return a JSON object:
```json
{sample_generated_scenario.model_json_schema()}
```
Here are the baseline PFun CMA model parameters, displayed as a markdown-formatted table:
{basal_param_descriptions}

Now consider a case when the user requests a non-baseline scenario-conditioned PFun CMA model parameters:
User: "a patient with chronic stress that exacerbates the risk of glucose lows in the evening"
Think: "Corresponding to the scenario, here is a hypothetical scenario-conditioned PFun CMA model parameters: "
{scenario_param_descriptions}
Assistant:
```json
{sample_generated_scenario.model_dump()}
```

Now, please generate a scenario based on the following user query. If the query is empty, generate a random scenario.
User: "{query if query else "No query provided."}"
Assistant:
"""
    # query the LLM with the formatted prompt, generate a scenario
    generated_scenario = await _call_llm_for_json(prompt, stream=stream)
    if "health_info" not in generated_scenario:
        # The query content is deliberately omitted from the log: WARNING logs
        # may be persisted/forwarded, and even a truncated preview can carry PII
        # (names, emails, phone numbers). Only the query length is logged to
        # keep the message actionable.
        logger.warning(
            "LLM response omitted 'health_info' (query length=%d); substituting the "
            "fixed sample health profile and marking used_fallback_health_info=True.",
            len(query) if query else 0,
        )
        generated_scenario["health_info"] = sample_health_info.model_dump()
        used_fallback_health_info = True
    else:
        used_fallback_health_info = False

    scenario = PFunLLMGeneratedScenario(**generated_scenario)
    scenario.used_fallback_health_info = used_fallback_health_info
    return scenario
