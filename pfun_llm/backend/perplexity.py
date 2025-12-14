"""Perplexity generative model interface."""
import logging
import os
from typing import Optional
from pydantic import BaseModel, Field, field_validator, field_serializer
from pfun_llm.backend.base import BaseGenerativeModel
import perplexity as pexai


class PerplexityMessage(BaseModel):
    """Message schema for Perplexity API."""
    role: str = Field(default="user")
    content: str = Field()


class PerplexityMessages(BaseModel):
    """Messages schema for Perplexity API."""
    messages: list[PerplexityMessage | str] = Field(default_factory=list)

    @field_serializer("messages")
    def serialize_messages(self, v):
        """Serialize messages to the format expected by the Perplexity API."""
        serialized_messages = []
        for message in v:
            if isinstance(message, PerplexityMessage):
                serialized_messages.append({
                    "role": message.role,
                    "content": message.content
                })
            else:
                raise ValueError(
                    "Each message must be a PerplexityMessage instance. "
                    "Received: ({}) {}".format(type(message), repr(message))
                )
        return serialized_messages

    @field_validator("messages", mode="before")
    @classmethod
    def validate_messages(cls, v):
        """Validate messages to ensure they are in the correct format."""
        if isinstance(v, list):
            validated_messages = []
            for item in v:
                if isinstance(item, str):
                    validated_messages.append(
                        PerplexityMessage(role="user", content=item))
                elif isinstance(item, PerplexityMessage):
                    validated_messages.append(item)
                else:
                    raise ValueError(
                        "Each message must be a string or a PerplexityMessage instance. "
                        "Received: ({}) {}".format(type(item), repr(item))
                    )
            return validated_messages
        else:
            raise ValueError("Messages must be provided as a list.")


class PerplexityGenerativeModel(BaseGenerativeModel):
    """Generative model interface for Perplexity LLM backend."""

    #: The default model to use if no model is specified.
    _default_model = "sonar-pro"

    def __new__(cls, *args, **kwargs):
        """Create a new instance of PerplexityGenerativeModel."""
        obj = super().__new__(cls, *args, **kwargs)
        obj._default_model = "sonar-pro"
        return obj

    def call_genai_client(
            self,
            model: Optional[str] = None,
            contents: Optional[list | str | PerplexityMessages | PerplexityMessage] = None):
        """Call the API client with the specified model and contents."""
        super().call_genai_client(model=model, contents=contents)
        if not isinstance(contents, PerplexityMessages):
            contents = PerplexityMessages(
                messages=contents if isinstance(contents, list) else [contents, ])
        serialized_messages = contents.model_dump()["messages"]
        logging.debug("Serialized messages for Perplexity API (type=%s): %s",
                      type(serialized_messages), repr(serialized_messages))
        response = self._client.chat.completions.create(
            model=model,
            messages=serialized_messages,
            web_search_options={"search_type": "pro"}
        )
        return response.choices[0].message.content

    @classmethod
    def setup_genai_client(cls) -> pexai.Perplexity:
        """Setup the Perplexity API client.
        Returns:
            pexai.Perplexity: The Perplexity API client.
        """
        # setup Perplexity client with API key from environment variable PERPLEXITY_API_KEY
        return pexai.Perplexity()
