"""Shared pytest fixtures for rag-eval tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class FakeMessage:
    """Minimal stand-in for an openai ChatCompletionMessage, enough for
    judges.py / synthetic_data.py to read `.content` off of.
    """

    content: str | None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeChatCompletion:
    """Minimal stand-in for an openai.types.chat.ChatCompletion, enough for
    judges.py / synthetic_data.py to extract `choices[0].message.content` from.
    """

    choices: list[FakeChoice] = field(default_factory=list)


def make_score_message(score: float, justification: str) -> FakeChatCompletion:
    payload = json.dumps({"score": score, "justification": justification})
    return FakeChatCompletion(choices=[FakeChoice(message=FakeMessage(content=payload))])


def make_raw_message(raw_text: str) -> FakeChatCompletion:
    return FakeChatCompletion(choices=[FakeChoice(message=FakeMessage(content=raw_text))])


def make_empty_choices_message() -> FakeChatCompletion:
    return FakeChatCompletion(choices=[])


def make_no_content_message() -> FakeChatCompletion:
    return FakeChatCompletion(choices=[FakeChoice(message=FakeMessage(content=None))])


class FakeCompletionsResource:
    """Stand-in for `client.chat.completions` that returns a queued sequence of responses."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def queue(self, response: Any) -> None:
        self._responses.append(response)

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeCompletionsResource.create called with no queued response")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@dataclass
class FakeChatResource:
    completions: FakeCompletionsResource


class FakeAsyncOpenAI:
    """Stand-in for `openai.AsyncOpenAI` exposing only `.chat.completions.create`."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.chat = FakeChatResource(completions=FakeCompletionsResource(responses))


@pytest.fixture
def fake_openai_client() -> FakeAsyncOpenAI:
    return FakeAsyncOpenAI()
