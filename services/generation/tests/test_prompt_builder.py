from __future__ import annotations

from rag_generation.prompt_builder import JSON_OUTPUT_INSTRUCTIONS, build_prompt
from tests.conftest import make_retrieved_chunk


class TestMessageShape:
    def test_returns_exactly_system_then_user_message(self) -> None:
        messages = build_prompt("What is the revenue?", [], "You are a helpful assistant.")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_message_content_is_plain_strings(self) -> None:
        # Unlike the old Anthropic content-block format, message content here
        # is a single string per message, not a list of typed blocks.
        messages = build_prompt("q", [], "sys")
        for message in messages:
            assert isinstance(message["content"], str)

    def test_system_prompt_text_appears_in_system_message(self) -> None:
        stable_prompt = "This is the exact stable system prompt text."
        messages = build_prompt("q", [], stable_prompt)
        assert stable_prompt in messages[0]["content"]

    def test_json_output_instructions_present_in_system_message(self) -> None:
        messages = build_prompt("q", [], "sys")
        assert JSON_OUTPUT_INSTRUCTIONS in messages[0]["content"]

    def test_json_instructions_mention_answer_and_citations_fields(self) -> None:
        messages = build_prompt("q", [], "sys")
        system_text = messages[0]["content"]
        assert '"answer"' in system_text
        assert '"citations"' in system_text
        assert "parent_id" in system_text
        assert "document_id" in system_text
        assert "page_number" in system_text


class TestOrdering:
    def test_volatile_content_is_in_user_message_not_system(self) -> None:
        chunk = make_retrieved_chunk("Confidential retrieved passage about revenue figures.")
        query = "What was the revenue?"
        messages = build_prompt(query, [chunk], "Stable system prompt.")

        system_text = messages[0]["content"]
        assert query not in system_text
        assert "revenue figures" not in system_text

        user_text = messages[1]["content"]
        assert query in user_text
        assert "revenue figures" in user_text

    def test_system_message_is_first_and_stable_across_queries(self) -> None:
        messages_a = build_prompt("question A", [], "Same stable system prompt.")
        messages_b = build_prompt("totally different question B", [], "Same stable system prompt.")
        # The system message (rendered first) is byte-identical across
        # different queries -- preserved as the "stable content first"
        # ordering principle, useful if the provider does any transparent
        # prefix caching, even though there's no explicit cache API to drive.
        assert messages_a[0] == messages_b[0]

    def test_user_message_varies_with_query(self) -> None:
        messages_a = build_prompt("question A", [], "Stable prompt.")
        messages_b = build_prompt("question B", [], "Stable prompt.")
        assert messages_a[1] != messages_b[1]
        assert messages_a[0] == messages_b[0]

    def test_no_cache_control_or_output_config_concepts_leak_into_messages(self) -> None:
        # The OpenAI-compatible API has no cache_control/output_config
        # mechanism at all; make sure none of that Anthropic-specific
        # machinery is present in the built messages.
        messages = build_prompt("q", [], "sys")
        for message in messages:
            assert "cache_control" not in message
            assert set(message.keys()) == {"role", "content"}


class TestContextRendering:
    def test_no_chunks_still_produces_a_valid_user_message(self) -> None:
        messages = build_prompt("What is X?", [], "sys")
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        assert "What is X?" in messages[1]["content"]

    def test_multiple_chunks_all_appear_in_rendered_context(self) -> None:
        chunk_a = make_retrieved_chunk("Alpha passage text.", parent_id="a", document_id="doc-a")
        chunk_b = make_retrieved_chunk("Beta passage text.", parent_id="b", document_id="doc-b")
        messages = build_prompt("q", [chunk_a, chunk_b], "sys")
        rendered = messages[1]["content"]
        assert "Alpha passage text." in rendered
        assert "Beta passage text." in rendered
        assert "parent_id=a" in rendered
        assert "parent_id=b" in rendered

    def test_no_chunks_renders_explicit_no_context_message(self) -> None:
        messages = build_prompt("q", [], "sys")
        assert "No relevant context passages were retrieved" in messages[1]["content"]
