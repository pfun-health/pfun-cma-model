import logging
import os
import json
import google.genai as genai
from pfun_cma_model.engine.cma_model_params import CMAModelParams


class GenerativeModel:
    def __init__(self, model='gemini-2.5-pro'):
        self._model = model
        self._client = self.setup_genai_client()

    def __call__(self, model=None, contents=None):
        if model is None:
            model = self._model
        if contents is None:
            contents = []
        if not isinstance(contents, list):
            contents = [contents, ]
        return self._client.models.generate_content(
            model=model,
            contents=contents
        )

    def generate_content(self, prompt: str):
        return self.__call__(model=self._model, contents=[prompt, ])

    @classmethod
    def setup_genai_client(cls):
        """Setup the Gemini API client.

        Returns:
            genai.Client: The Gemini API client.
        """
        try:
            gemini_api_key_or_path = os.environ["GEMINI_API_KEY"]
            if os.path.isfile(gemini_api_key_or_path):
                with open(gemini_api_key_or_path, "r") as f:
                    gemini_api_key = f.read().strip()
            else:
                gemini_api_key = gemini_api_key_or_path
        except KeyError:
            raise Exception("GEMINI_API_KEY environment variable not set.")
        # (only for DEBUG, print the api key)
        logging.debug("Gemini API key: %s", gemini_api_key)
        client = genai.Client(api_key=gemini_api_key)
        return client


def translate_query_to_params(query: str) -> dict:
    """
    Translates a plain English query into PFun CMA model parameters using the Gemini API.

    Args:
        query: The plain English query.

    Returns:
        A dictionary containing the PFun CMA model parameters.
    """
    model = GenerativeModel()

    # Construct the prompt
    params = CMAModelParams()
    param_descriptions = params.generate_markdown_table()

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

    response = model.generate_content(prompt)

    # Extract the JSON from the response
    try:
        # The response might contain markdown, so we need to extract the JSON from it
        json_str = response.text.strip().replace("`", "").replace("json", "")
        params = json.loads(json_str)
        return params
    except (json.JSONDecodeError, KeyError) as e:
        raise Exception(f"Failed to parse Gemini API response: {e}") from e


def generate_causal_explanation(description: str, trace: str) -> dict:
    """
    Generates a causal explanation for a glucose trace given a patient description.

    Args:
        description: A narrative describing the person's health, lifestyle, and recent events.
        trace: A JSON string representing the glucose trace data.

    Returns:
        A dictionary containing the causal explanation.
    """

    model = GenerativeModel()

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

    response = model.generate_content(prompt)

    try:
        json_str = response.text.strip().replace("`", "").replace("json", "")
        explanation = json.loads(json_str)
        return explanation
    except (json.JSONDecodeError, KeyError) as e:
        raise Exception(f"Failed to parse Gemini API response: {e}") from e


def generate_scenario(query: str = None) -> dict:
    """
    Generates a realistic "pfun-scene" JSON object using the Gemini API.

    Args:
        query: An optional query to guide the scenario generation.

    Returns:
        A dictionary containing the generated scenario.
    """

    model = GenerativeModel()

    # Construct the prompt
    params = CMAModelParams()
    param_descriptions = params.generate_markdown_table()

    prompt = f"""\
You are a helpful assistant that generates realistic scenarios for a person with diabetes.

The user may provide a query to guide the generation, or you can create a scenario from scratch.

You will return a JSON object with the following structure:
{{
    "qualitative_description": "A narrative describing the person's health, lifestyle, and recent events.",
    "parameters": {{
        "param1": value1,
        "param2": value2,
        ...
    }}
}}

Here are the PFun CMA model parameters and their descriptions:
{param_descriptions}

Here is an example:
User: "a patient with chronic stress that exacerbates the risk of glucose lows in the evening"
Assistant:
```json
{{
    "qualitative_description": "This individual is experiencing a period of high stress due to work deadlines, which has been disrupting their sleep patterns and leading to poor dietary choices, especially in the evenings. They often skip meals during the day and then have a large, carbohydrate-heavy dinner late at night. This, combined with the physiological effects of stress, has increased their risk of nocturnal hypoglycemia.",
    "parameters": {{
        "Cm": 1.5,
        "B": -0.2
    }}
}}
```

Now, please generate a scenario based on the following user query. If the query is empty, generate a random scenario.
User: "{query if query else 'No query provided.'}"
Assistant:
"""

    response = model.generate_content(prompt)

    # Extract the JSON from the response
    try:
        # The response might contain markdown, so we need to extract the JSON from it
        json_str = response.text.strip().replace("`", "").replace("json", "")
        scenario = json.loads(json_str)
        return scenario
    except (json.JSONDecodeError, KeyError) as e:
        raise Exception(f"Failed to parse Gemini API response: {e}") from e
