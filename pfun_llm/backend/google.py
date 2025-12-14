import logging
import os
from typing import Optional
import google.genai as genai
from pfun_llm.backend.base import BaseGenerativeModel


class GeminiGenerativeModel(BaseGenerativeModel):
    """Generative model interface for Gemini (Google Vertex AI backend)."""

    #: The default model to use if no model is specified.
    _default_model = "gemini-2.5-flash"

    def call_genai_client(
            self,
            model: Optional[str] = None,
            contents: Optional[list | str] = None):
        """Call the API client with the specified model and contents."""
        super().call_genai_client(model=model, contents=contents)
        return self._client.models.generate_content(
            model=model,
            contents=contents
        )

    @classmethod
    def setup_genai_client(cls) -> genai.Client:
        """Setup the Gemini API client.

        Returns:
            genai.Client: The Gemini API client.
        """
        # use VertexAI with auth ADC
        client = genai.Client(
            vertexai=True,
            project=os.environ.get(
                "GOOGLE_CLOUD_PROJECT_ID", "pfun-cma-model"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        )
        logging.debug("Gemini API client setup successfully.")
        logging.debug("Gemini API client: %s", repr(client))
        return client
