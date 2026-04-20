"""Tests for backend output parity.

Verifies that the same prompt through FakeLLMBackend produces identical
parsed output from both annotation code paths (validating that the
parse/validate pipeline is deterministic).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_diagnostics.llm_annotator import validate_categories
from tests.fake_llm_backend import FakeLLMBackend


class TestBackendParity:
    """Same prompt through FakeLLMBackend produces identical parsed output."""

    @pytest.mark.parametrize(
        "prompt_text",
        [
            "Trial with retrieval_failure in trajectory",
            "Trial showing query_churn behavior",
            "Trial with clean_success outcome",
            "Trial exhibiting over_exploration pattern",
            "Unknown prompt with no matching category",
        ],
        ids=[
            "retrieval_failure",
            "query_churn",
            "clean_success",
            "over_exploration",
            "fallback",
        ],
    )
    def test_same_prompt_identical_output(self, prompt_text: str) -> None:
        """Two independent FakeLLMBackend instances return the same output."""
        backend_a = FakeLLMBackend()
        backend_b = FakeLLMBackend()

        result_a = backend_a.annotate(prompt_text)
        result_b = backend_b.annotate(prompt_text)

        assert result_a == result_b

        # Validate through the same validation pipeline
        validated_a = validate_categories(result_a["categories"], Path("/tmp/fake-trial"))
        validated_b = validate_categories(result_b["categories"], Path("/tmp/fake-trial"))

        assert validated_a == validated_b

    def test_deterministic_across_calls(self) -> None:
        """Same backend, same prompt, two calls produce identical categories."""
        backend = FakeLLMBackend()
        prompt = "Trial with edit_verify_loop_failure detected"

        result1 = backend.annotate(prompt)
        result2 = backend.annotate(prompt)

        assert result1 == result2
        assert backend.call_count == 2
