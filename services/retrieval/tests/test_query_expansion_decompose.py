"""Tests for `QueryExpander.decompose` (ADR-025 multi-hop query decomposition).

The Groq client is faked with a minimal stand-in exposing exactly
`chat.completions.create(...)`, returning a scripted response — the actual
Groq/HTTP wiring is exercised by `_complete`'s existing retry decorator and
covered indirectly through the multi_query/hyde paths already in production
use; what's new here is decompose's own parsing/dedup/fallback contract.
"""

from __future__ import annotations

import pytest

from rag_retrieval.query_expansion import QueryExpander, QueryExpansionError


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._reply)


class _FakeChat:
    def __init__(self, reply: str) -> None:
        self.completions = _FakeCompletions(reply)


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self.chat = _FakeChat(reply)


def _expander(reply: str) -> tuple[QueryExpander, _FakeClient]:
    client = _FakeClient(reply)
    return QueryExpander(client, model="test-utility-model"), client


@pytest.mark.asyncio
class TestDecomposeMultiHop:
    async def test_multi_hop_question_returns_original_plus_subquestions(self) -> None:
        expander, _ = _expander(
            "What was Acme Corp's Q3 revenue?\nWhat was Beta Inc's Q3 revenue?"
        )

        result = await expander.decompose("How does Acme's Q3 revenue compare to Beta's?")

        assert result[0] == "How does Acme's Q3 revenue compare to Beta's?"
        assert "What was Acme Corp's Q3 revenue?" in result
        assert "What was Beta Inc's Q3 revenue?" in result
        assert len(result) == 3

    async def test_original_query_is_always_first(self) -> None:
        """Sub-questions supplement the direct hit, never replace it — a bad
        decomposition can only ADD candidates pre-fusion, not remove the
        query that would have worked on its own."""
        expander, _ = _expander("sub-question A\nsub-question B")

        result = await expander.decompose("original query")

        assert result[0] == "original query"


@pytest.mark.asyncio
class TestDecomposeSingleFactNoOp:
    async def test_none_reply_returns_only_the_original_query(self) -> None:
        """A single-fact lookup — the model correctly identifies no
        decomposition is needed — must be a safe no-op, not an empty list
        or an error."""
        expander, _ = _expander("NONE")

        result = await expander.decompose("What is Acme Corp's ticker symbol?")

        assert result == ["What is Acme Corp's ticker symbol?"]

    async def test_none_reply_is_case_insensitive(self) -> None:
        expander, _ = _expander("none")

        result = await expander.decompose("simple lookup")

        assert result == ["simple lookup"]

    async def test_none_reply_with_surrounding_whitespace_still_detected(self) -> None:
        expander, _ = _expander("  NONE  \n")

        result = await expander.decompose("simple lookup")

        assert result == ["simple lookup"]


@pytest.mark.asyncio
class TestDecomposeParsingAndDedup:
    async def test_strips_numbering_and_bullets_from_subquestions(self) -> None:
        expander, _ = _expander("1. What is X?\n2) What is Y?\n- What is Z?")

        result = await expander.decompose("original")

        assert "What is X?" in result
        assert "What is Y?" in result
        assert "What is Z?" in result

    async def test_deduplicates_case_insensitively_against_original(self) -> None:
        """A sub-question that's just the original query restated (case-
        insensitively) must not appear twice."""
        expander, _ = _expander("What is Acme's revenue?\nWhat is Beta's revenue?")

        result = await expander.decompose("what is acme's revenue?")

        # Original preserved verbatim as given; the model's restatement of
        # the same fact is dropped as a duplicate.
        assert result.count("What is Beta's revenue?") == 1
        assert not any(
            r.lower() == "what is acme's revenue?" and r != "what is acme's revenue?"
            for r in result[1:]
        )

    async def test_blank_lines_are_skipped(self) -> None:
        expander, _ = _expander("What is X?\n\n\nWhat is Y?")

        result = await expander.decompose("original")

        assert result == ["original", "What is X?", "What is Y?"]

    async def test_respects_max_subquestions_cap(self) -> None:
        expander, _ = _expander("q1\nq2\nq3\nq4\nq5\nq6")

        result = await expander.decompose("original", max_subquestions=2)

        assert len(result) == 3  # original + 2, not + 6
        assert result == ["original", "q1", "q2"]

    async def test_empty_reply_yields_just_the_original(self) -> None:
        expander, _ = _expander("")

        with pytest.raises(QueryExpansionError):
            await expander.decompose("original")
