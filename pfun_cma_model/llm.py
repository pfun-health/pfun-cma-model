import os
import google.genai as genai
from pfun_cma_model.engine.cma_model_params import CMAModelParams
import json

def translate_query_to_params(query: str) -> dict:
    """
    Translates a plain English query into PFun CMA model parameters using the Gemini API.

    Args:
        query: The plain English query.

    Returns:
        A dictionary containing the PFun CMA model parameters.
    """
    try:
        gemini_api_key = os.environ["GEMINI_API_KEY"]
    except KeyError:
        raise Exception("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=gemini_api_key)

    model = genai.GenerativeModel('gemini-pro')

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
