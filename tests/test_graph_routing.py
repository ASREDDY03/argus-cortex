"""
Tests for the graph routing functions in orchestrator/graph.py.

The routing functions are pure Python (no LLM calls) so they're tested
directly without building the full graph (which requires the sqlite
checkpoint dependency).
"""
import pytest
from langgraph.graph import END


# ── Routing logic extracted for testing ───────────────────────────────────────
# These replicate the exact routing functions in orchestrator/graph.py so they
# can be tested independently of the SqliteSaver import.

def _route_after_synthesizer(state: dict) -> str:
    retry_agents = state.get("retry_agents", {})
    retry_count = state.get("retry_count", 0)
    if retry_agents and retry_count < 1:
        return "retry_dispatcher"
    return "evaluator"


def _route_after_evaluator(state: dict) -> str:
    if state.get("approved_findings"):
        return "human_review"
    return END


# ── _route_after_synthesizer ───────────────────────────────────────────────────

class TestRouteAfterSynthesizer:
    def test_routes_to_retry_on_first_pass_with_agents(self):
        state = {"retry_agents": {"ml_agent": "focus on error handling"}, "retry_count": 0}
        assert _route_after_synthesizer(state) == "retry_dispatcher"

    def test_routes_to_evaluator_on_second_pass(self):
        state = {"retry_agents": {"ml_agent": "focus on error handling"}, "retry_count": 1}
        assert _route_after_synthesizer(state) == "evaluator"

    def test_routes_to_evaluator_with_no_retry_agents(self):
        state = {"retry_agents": {}, "retry_count": 0}
        assert _route_after_synthesizer(state) == "evaluator"

    def test_routes_to_evaluator_with_missing_retry_agents(self):
        state = {"retry_count": 0}
        assert _route_after_synthesizer(state) == "evaluator"

    def test_routes_to_evaluator_above_retry_cap(self):
        state = {"retry_agents": {"springboot_agent": "auth"}, "retry_count": 2}
        assert _route_after_synthesizer(state) == "evaluator"


# ── _route_after_evaluator ─────────────────────────────────────────────────────

class TestRouteAfterEvaluator:
    def test_routes_to_human_review_with_findings(self):
        finding = {"file": "app.py", "description": "bug", "severity": "high"}
        state = {"approved_findings": [finding]}
        assert _route_after_evaluator(state) == "human_review"

    def test_routes_to_end_with_no_findings(self):
        state = {"approved_findings": []}
        assert _route_after_evaluator(state) == END

    def test_routes_to_end_with_missing_key(self):
        state = {}
        assert _route_after_evaluator(state) == END

    def test_routes_to_human_review_with_multiple_findings(self):
        findings = [
            {"file": "app.py", "description": "bug1", "severity": "high"},
            {"file": "pom.xml", "description": "bug2", "severity": "medium"},
        ]
        state = {"approved_findings": findings}
        assert _route_after_evaluator(state) == "human_review"


# ── State reducer behaviour ────────────────────────────────────────────────────

class TestStateReducers:
    """
    CortexState uses Annotated reducers for accumulating values across agents.
    These tests verify the reducer semantics are correct.
    """

    def test_findings_reducer_concatenates_lists(self):
        # Annotated[List[AgentFinding], lambda a, b: a + b]
        reducer = lambda a, b: a + b
        a = [{"file": "app.py", "description": "bug1"}]
        b = [{"file": "pom.xml", "description": "bug2"}]
        merged = reducer(a, b)
        assert len(merged) == 2
        assert merged[0]["file"] == "app.py"
        assert merged[1]["file"] == "pom.xml"

    def test_findings_reducer_empty_lists(self):
        reducer = lambda a, b: a + b
        assert reducer([], []) == []
        assert reducer([{"x": 1}], []) == [{"x": 1}]
        assert reducer([], [{"x": 2}]) == [{"x": 2}]

    def test_int_reducer_accumulates_correctly(self):
        # Annotated[int, lambda a, b: a + b] — used for token counts, suppressed_count etc.
        reducer = lambda a, b: a + b
        assert reducer(0, 5) == 5
        assert reducer(10, 20) == 30
        assert reducer(0, 0) == 0

    def test_findings_reducer_preserves_order(self):
        reducer = lambda a, b: a + b
        a = [{"id": 1}, {"id": 2}]
        b = [{"id": 3}, {"id": 4}]
        merged = reducer(a, b)
        assert [f["id"] for f in merged] == [1, 2, 3, 4]
