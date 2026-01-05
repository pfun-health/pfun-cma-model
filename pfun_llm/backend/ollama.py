"""Ollama-backend class for generative model interfaces."""
import asyncio
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, field_serializer
from ollama import AsyncClient
from pfun_common.settings import get_settings


class OllamaMessage(BaseModel):
    """Message schema for Ollama API."""
    role: str = Field(default="user")
    content: str = Field()


class OllamaMessages(BaseModel):
    """Messages schema for Ollama API."""
    messages: list[OllamaMessage | str] = Field(default_factory=list)

    @field_serializer("messages")
    def serialize_messages(self, v):
        """Serialize messages to the format expected by the Ollama API."""
        serialized_messages = []
        for message in v:
            if isinstance(message, OllamaMessage):
                serialized_messages.append({
                    "role": message.role,
                    "content": message.content
                })
            else:
                raise ValueError(
                    "Each message must be a OllamaMessage instance. "
                    "Received: ({}) {}".format(type(message), repr(message))
                )
        return serialized_messages


_OLLAMA_DEFAULT_MODEL: Literal["tinyllama"] = "tinyllama"


class OllamaGenerativeModel:
    """Ollama-backend class for generative model interfaces."""

    #: The default model to use if no model is specified.
    _default_model = _OLLAMA_DEFAULT_MODEL

    def __new__(cls, *args, **kwargs):
        """Create a new instance of OllamaGenerativeModel."""
        obj = super().__new__(cls, *args, **kwargs)
        obj._default_model = _OLLAMA_DEFAULT_MODEL
        return obj

    async def stream_chat(self, messages, model=None):
        if model is None:
            model = self._model
        async for part in await self._client.chat(model=model, messages=messages, stream=True):
            yield part['message']['content']

    async def chat(self, messages, model=None):
        if model is None:
            model = self._model
        response = await self._client.chat(
            model=model,
            messages=messages
        )
        return response

    def call_genai_client(
            self,
            model: Optional[str] = None,
            contents: Optional[list | str | OllamaMessages | OllamaMessage] = None,
            stream: bool = False):
        """Call the API client with the specified model and contents."""
        super().call_genai_client(model=model, contents=contents)
        if not isinstance(contents, OllamaMessages):
            contents = OllamaMessages(
                messages=contents if isinstance(contents, list) else [contents, ])
        serialized_messages = contents.model_dump()["messages"]
        logging.debug("Serialized messages for Ollama API (type=%s): %s",
                      type(serialized_messages), repr(serialized_messages))
        # ensure the response is an awaitable (avoid making this method async, handle in context)
        response_future = asyncio.ensure_future(
            self.chat(model=model, messages=serialized_messages)
        )
        return response_future

    @classmethod
    def setup_genai_client(cls) -> AsyncClient:
        """Setup the API client for ollama backend.

        Returns:
            ollama.AsyncClient: The ollama API client.
        """
        settings = get_settings()
        client = AsyncClient(
            host=settings.ollama_host
        )
        logging.debug("Ollama API client setup successfully.")
        logging.debug("Ollama API client: %s", repr(client))
        return client
