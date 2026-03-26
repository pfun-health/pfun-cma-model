"""Ollama-backend class for generative model interfaces."""

import logging
import asyncio
import importlib
from typing import Optional, Literal, Any
import ollama
from pydantic import BaseModel, Field, field_serializer
from ollama import (
    AsyncClient,
    ProgressResponse,
)
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


OllamaDefaultModel = Literal[
    "gpt-oss:120b-cloud", "gemma3:4b-cloud", "deepseek-v3.2:cloud"
]

_OLLAMA_DEFAULT_MODEL: OllamaDefaultModel = "gpt-oss:120b-cloud"


def _conv_str2msg(
    message_content: str | OllamaMessage, role: str = "user"
) -> OllamaMessage:
    """convert raw string to OllamaMessage."""
    if isinstance(message_content, OllamaMessage):
        return message_content
    return OllamaMessage(content=message_content, role=role)


def _format_messages(raw_messages: str | list, role: str = "user") -> OllamaMessages:  # type: ignore
    """Format raw messages (str|list), return OllamaMessages object."""

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
        get_settings = importlib.import_module("pfun_common.settings").get_settings  # type: ignore
        settings = get_settings()
        obj._default_model = kwargs.get("model", settings.ollama_model)
        return obj

    def __init__(self, model: str | None = None, **kwargs):
        super().__init__(model, **kwargs)
        # Set the default model for this instance (use settings)
        logging.debug(f"OllamaGenerativeModel initialized with model: {self._model}")

    async def stream_chat(self, messages, model=None):
        if model is None:
            model = self._model
        async for part in await self._client.chat(
            model=model,
            messages=messages,
            stream=True,
            **self._extra_kwds,  # with streaming enabled
        ):
            yield part["message"]["content"]

    async def chat(self, messages, model=None):
        if model is None:
            model = self._model
        logging.debug("Extra arguments (called on ollama.chat): %s", self._extra_kwds)
        response = await asyncio.ensure_future(
            self._client.chat(model=model, messages=messages, **self._extra_kwds)
        )
        return response

    def call_genai_client(
        self,
        model: Optional[str] = None,
        contents: Optional[list | str | OllamaMessages | OllamaMessage] = None,
        stream: bool = True,
        **kwds,
    ) -> None | Any | asyncio.Future:
        """Call the API client with the specified model and contents.

        :param: stream [bool] : flag to indicate streaming for the agent chat call.
        """
        model = super().call_genai_client(model=model, contents=contents, **kwds) # type: ignore
        if not isinstance(contents, OllamaMessages):
            contents = _format_messages(contents)  # type: ignore
        serialized_messages = contents.model_dump()["messages"]
        logging.debug(
            "Serialized messages for Ollama API (type=%s): %s",
            type(serialized_messages),
            repr(serialized_messages),
        )
        try:
            # ensure the response is an awaitable (avoid making this method async, handle in context)
            return self.chat(model=model, messages=serialized_messages)
        except ollama._types.ResponseError as exc:  # type: ignore
            logging.error("Ollama API error: %s", exc)
            if exc.status_code == 404:
                logging.warning(
                    "Model not found: %s. Please check the model name and ensure it is available in your Ollama instance. Now attempting to pull the specified model.",
                    model,
                )
                prog_response: ProgressResponse = ollama.pull(model)  # type: ignore
                logging.info("Pull response: %s", prog_response)
                if prog_response.status.lower() == "completed":  # type: ignore
                    logging.info("Model pulled successfully. Retrying the API call.")
                    return self.chat(model=model, messages=serialized_messages)
            # if we reach here, it means the error was not a 404 or the pull did not succeed, so we raise the error
            raise RuntimeError(f"Ollama API error: {exc}") from exc

    @classmethod
    def setup_genai_client(cls) -> AsyncClient:
        """Setup the API client for ollama backend.

        Returns:
            ollama.AsyncClient: The ollama API client.
        """
        get_settings = importlib.import_module("pfun_common.settings").get_settings  # type: ignore
        settings = get_settings()
        client = AsyncClient(host=settings.ollama_host)
        logging.debug("Ollama API client setup successfully.")
        logging.debug("Ollama API client: %s", repr(client))
        return client
