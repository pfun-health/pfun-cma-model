"""OpenAI-backend class for generative model interfaces."""
from typing import Optional
from openai import OpenAI
from pfun_common.setttings import get_settings


class OpenaiGenerativeModel:
    """OpenAI-backend class for generative model interfaces."""

    #: The default model to use if no model is specified.
    _default_model = "gpt4o"

    def call_genai_client(self, model: Optional[str] = None, contents: Optional[list | str] = None):
        """Call the API client with the specified model and contents."""
        super().call_genai_client(model=model, contents=contents)
        return self._client.responses.create(
            model=model,
            # instructions=...
            input=contents
        )

    @classmethod
    def setup_genai_client(cls):
        """Setup the API client.

        Returns:
            genai.Client: The API client.
        """
        settings = get_settings()
        client = OpenAI(
            api_key=settings.openai_key
        )
        logging.debug("OpenAI API client setup successfully.")
        logging.debug("OpenAI API client: %s", repr(client))
        return client
            
