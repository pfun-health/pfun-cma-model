"""Base class for generative model interfaces."""

from typing import Optional, Any


class BaseGenerativeModel:
    """Base class for generative model interfaces."""

    #: The default model to use if no model is specified.
    _default_model: Optional[str] = None

    def __init__(self, model: Optional[str] = None, **kwargs):
        self._model = model if model is not None else self.__class__._default_model
        self._client = self.setup_genai_client()
        self._extra_kwds = kwargs or {}

    def __call__(
        self,
        model: Optional[str] = None,
        contents: Optional[list | str] = None,
        **kwargs,
    ) -> None | Any:
        """Call the API with the specified model and contents."""
        if model is None:
            model = self._model
        if contents is None:
            raise ValueError("Contents must be provided.")
        return self.call_genai_client(model=model, contents=contents, **kwargs)

    def generate_content(self, prompt: str, **kwargs) -> None | Any:
        """(convenience method) Generate content using the specified prompt."""
        return self(model=self._model, contents=prompt, **kwargs)

    def call_genai_client(
        self, model: Optional[str] = None, contents: Optional[list | str] = None, **kwds
    ) -> None | Any:
        """Call the API client with the specified model and contents."""
        if model is None:
            model = self._model
        return model

    @classmethod
    def setup_genai_client(cls):
        """Setup the API client.

        Returns:
            genai.Client: The API client.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
