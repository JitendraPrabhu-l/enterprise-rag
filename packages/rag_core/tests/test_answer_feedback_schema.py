"""Tests for `AnswerFeedback` (ADR-027) — the schema's own validation
contract. Endpoint-level behavior (metric increment, structured logging)
is covered in services/generation/tests/test_feedback_endpoint.py.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from rag_core.schemas import AnswerFeedback


class TestRatingValidation:
    def test_up_is_valid(self) -> None:
        fb = AnswerFeedback(request_id=uuid4(), rating="up")
        assert fb.rating == "up"

    def test_down_is_valid(self) -> None:
        fb = AnswerFeedback(request_id=uuid4(), rating="down")
        assert fb.rating == "down"

    def test_any_other_string_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnswerFeedback(request_id=uuid4(), rating="neutral")

    def test_uppercase_is_rejected_not_silently_normalized(self) -> None:
        """No case-folding: a client sending 'Up' is a client bug worth
        surfacing as a 422, not a value worth guessing at."""
        with pytest.raises(ValidationError):
            AnswerFeedback(request_id=uuid4(), rating="Up")

    def test_empty_string_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnswerFeedback(request_id=uuid4(), rating="")


class TestOptionalContextFields:
    def test_query_answer_comment_all_default_to_none(self) -> None:
        fb = AnswerFeedback(request_id=uuid4(), rating="down")
        assert fb.query is None
        assert fb.answer is None
        assert fb.comment is None

    def test_optional_fields_are_accepted_when_provided(self) -> None:
        fb = AnswerFeedback(
            request_id=uuid4(),
            rating="down",
            query="What was Q3 revenue?",
            answer="I don't know.",
            comment="This should have been in the filing.",
        )
        assert fb.query == "What was Q3 revenue?"
        assert fb.comment == "This should have been in the filing."
