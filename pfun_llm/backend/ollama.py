"""Ollama-backend class for generative model interfaces."""

import logging
import asyncio
from typing import Optional, Literal, Any
from pydantic import BaseModel, Field, field_serializer
from ollama import AsyncClient
from pfun_common.settings import get_settings
from pfun_llm.backend.base import BaseGenerativeModel


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
                serialized_messages.append(
                    {"role": message.role, "content": message.content}
                )
            else:
                raise ValueError(
                    "Each message must be a OllamaMessage instance. "
                    "Received: ({}) {}".format(type(message), repr(message))
                )
        return serialized_messages


OllamaDefaultModel = Literal["gpt-oss:120b-cloud", "gemma3:4b-cloud"]

_OLLAMA_DEFAULT_MODEL: OllamaDefaultModel = "gpt-oss:120b-cloud"


def _conv_str2msg(
    message_content: str | OllamaMessage, role: str = "user"
) -> OllamaMessage:
    """convert raw string to OllamaMessage."""
    if isinstance(message_content, OllamaMessage):
        return message_content
    return OllamaMessage(content=message_content, role=role)


def _format_messages(
        raw_messages: str | list,  # type: ignore
        role: str = "user"
) -> OllamaMessages:
    """format raw messages (str|list), return OllamaMessages object."""

    if not isinstance(raw_messages, list):
        raw_messages: list = [
            raw_messages,
        ]
    return OllamaMessages(
        messages=[_conv_str2msg(msg_, role=role) for msg_ in raw_messages]
    )


class OllamaGenerativeModel(BaseGenerativeModel):
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
        async for part in await self._client.chat(
            model=model, messages=messages, stream=True
        ):
            yield part["message"]["content"]

    async def chat(self, messages, model=None):
        if model is None:
            model = self._model
        response = await asyncio.ensure_future(
            self._client.chat(model=model, messages=messages)
        )
        return response

    async def call_genai_client(
        self,
        model: Optional[str] = None,
        contents: Optional[list | str | OllamaMessages | OllamaMessage] = None,
        **kwds,
    ) -> None | Any | asyncio.Future:
        """Call the API client with the specified model and contents."""
        super().call_genai_client(model=model, contents=contents, **kwds)  # type: ignore
        if not isinstance(contents, OllamaMessages):
            contents = _format_messages(contents)  # type: ignore
        serialized_messages = contents.model_dump()["messages"]
        logging.debug(
            "Serialized messages for Ollama API (type=%s): %s",
            type(serialized_messages),
            repr(serialized_messages),
        )
        # ensure the response is an awaitable (avoid making this method async, handle in context)
        return self.chat(model=model, messages=serialized_messages)

    @classmethod
    def setup_genai_client(cls) -> AsyncClient:
        """Setup the API client for ollama backend.

        Returns:
            ollama.AsyncClient: The ollama API client.
        """
        settings = get_settings()
        client = AsyncClient(host=settings.ollama_host)
        logging.debug("Ollama API client setup successfully.")
        logging.debug("Ollama API client: %s", repr(client))
        return client
