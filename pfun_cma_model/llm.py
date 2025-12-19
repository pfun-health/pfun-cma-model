import logging
import os
import json
import re
from typing import Optional, Any
from pfun_llm.backend.perplexity import PerplexityGenerativeModel as GenerativeModel
from pfun_cma_model.engine.cma_model_params import CMAModelParams


def _get_resp_text(response: Any | str) -> str:
    """Get the response text attribute if it exists, otherwise return the string."""
    return str(getattr(response, "text", str(response)))


def _call_llm_for_json(prompt: str) -> dict:
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
    resp_text: str = _get_resp_text(response)
    try:
        # The response might contain markdown, so we need to extract the JSON from it
        json_match = re.search(
            r"```json\s*([\s\S]*?)\s*```", resp_text, re.DOTALL)
        json_str = json_match.group(1) if json_match else resp_text.strip().replace(
            "`", "").replace("json", "")
        return json.loads(json_str)
    except (json.JSONDecodeError, KeyError, AttributeError, IndexError) as e:
        raise Exception(f"Failed to parse Gemini API response: {e}") from e


def translate_query_to_params(query: str) -> dict:
    """
    Translates a plain English query into PFun CMA model parameters using the Gemini API.

    Args:
        query: The plain English query.

    Returns:
        A dictionary containing the PFun CMA model parameters.
    """
    # Construct the prompt
    params = CMAModelParams()
    param_descriptions = params.generate_markdown_table(output_fmt="md")

    prompt = f"""\
You are a helpful assistant that translates plain English descriptions of a person's health into PFun CMA model parameters.

The user will provide a description, and you will return a JSON object with the corresponding model parameters.

Here are the PFun CMA model parameters and their descriptions:
{param_descriptions}

Here is an example:
User: "a patient with chronic stress that exacerbates the risk of glucose lows in the evening"
Assistant:
```json
{{
    "Cm": 1.5,
    "B": -0.2
}}
```

Now, please translate the following user query into PFun CMA model parameters.
User: "{query}"
Assistant:
"""
    return _call_llm_for_json(prompt)


def generate_causal_explanation(description: str, trace: str) -> dict:
    """
    Generates a causal explanation for a glucose trace given a patient description.

    Args:
        description: A narrative describing the person's health, lifestyle, and recent events.
        trace: A JSON string representing the glucose trace data.

    Returns:
        A dictionary containing the causal explanation.
    """

    prompt = f"""\
You are a helpful assistant that analyzes glucose data for a person with diabetes and provides a causal explanation for the observed patterns.

You will be given a description of the person's health and lifestyle, and a JSON object representing their glucose trace over time.

You will return a JSON object with a single key, "causal_explanation", which is a list of potential actions and their probabilities of being the cause for the observed glucose pattern. The probabilities should be realistic and proportionate to the actual likelihood of each action.

Here is an example:
Description: "This individual is experiencing a period of high stress due to work deadlines, which has been disrupting their sleep patterns and leading to poor dietary choices, especially in the evenings. They often skip meals during the day and then have a large, carbohydrate-heavy dinner late at night. This, combined with the physiological effects of stress, has increased their risk of nocturnal hypoglycemia."
Trace: {{ ... (json data of glucose trace) ... }}
Assistant:
```json
{{
    "causal_explanation": [
        {{"action": "Ate a large, high-carb meal late at night", "probability": 0.6}},
        {{"action": "Experienced high stress", "probability": 0.3}},
        {{"action": "Inconsistent sleep schedule", "probability": 0.1}}
    ]
}}
```

Now, please analyze the following description and glucose trace and provide a causal explanation.
Description: "{description}"
Trace: {trace}
Assistant:
"""

    response = GenerativeModel().generate_content(prompt)

    try:
        json_str = _get_resp_text(response).strip().replace(
            "`", "").replace("json", "")  # type: ignore
        explanation = json.loads(json_str)
        return explanation
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        raise Exception(f"Failed to parse LLM API response: {e}") from e


def generate_scenario(query: Optional[str] = None) -> dict:
    """
    Generates a realistic "pfun-scene" JSON object using the Gemini API.

    Args:
        query: An optional query to guide the scenario generation.

    Returns:
        A dictionary containing the generated scenario.
    """

    # Construct the prompt

    # baseline cma model parameters
    basal_params = CMAModelParams()
    basal_param_descriptions = basal_params.generate_markdown_table(output_fmt="md")

    # hypothetical scenario-conditioned parameters
    scenario_description = (
        "This individual is experiencing a period of high stress ($C_m >> 0.0$) due to work deadlines, "
        "which has been disrupting their sleep patterns and leading to poor dietary choices, especially in the evenings. "
        "Their diet lacks high-quality proteins & fats, so their endogenous glucose production is dangerously unreliable ($B << 0.05$). "
        "Combined with the physiological effects of stress, they have an increased risk of experiencing episodes of nocturnal "
        "hypoglycemia, i.e. dangerously low blood glucose levels."
    )
    scenario_params = CMAModelParams(Cm=1.5, B=-0.2)
    scenario_param_descriptions = scenario_params.generate_markdown_table(
        output_fmt="md",
        included_params=["Cm", "B"],  # Only include the parameters that are different from the baseline
    )

    prompt = f"""\
You are a helpful assistant that generates realistic scenarios for a person with diabetes.

The user may provide a query to guide the generation, or you can create a scenario from scratch.

You will return a JSON object with the following structure:
```json
{{
    "qualitative_description": "A narrative describing the person's health, lifestyle, and recent events.",
    "parameters": {{
        "param1": {{ "value": value1, "description": "Description of param1" }},
        "param2": {{ "value": value2, "description": "Description of param2" }},
        ...
    }}
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
    "qualitative_description": "{scenario_description}",
    "parameters": {{
        "Cm": {{ "value": 1.5,  "stderr": 0.5, "description": "Heightened stress level, leading to increased cortisol-mediated glucose variability" }},
        "B": {{ "value": -0.2, "stderr": 0.25, "description": "Low baseline glucose" }}
    }}
}}
```

Now, please generate a scenario based on the following user query. If the query is empty, generate a random scenario.
User: "{query if query else 'No query provided.'}"
Assistant:
"""
    return _call_llm_for_json(prompt)


if __name__ == "__main__":
    # Example usage
    query = "a patient with chronic stress that exacerbates the risk of glucose lows in the evening"
    params = translate_query_to_params(query)
    print("Translated Parameters:", params)

    description = "This individual is experiencing a period of high stress due to work deadlines, which has been disrupting their sleep patterns and leading to poor dietary choices, especially in the evenings. They often skip meals during the day and then have a large, carbohydrate-heavy dinner late at night. This, combined with the physiological effects of stress, has increased their risk of nocturnal hypoglycemia."
    trace = '{"glucose_readings": [150, 140, 130, 120, 110, 100, 90, 80, 70, 60]}'
    explanation = generate_causal_explanation(description, trace)
    print("Causal Explanation:", explanation)

    scenario = generate_scenario(query)
    print("Generated Scenario:", scenario)