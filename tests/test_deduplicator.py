"""
Tests for orchestrator/deduplicator.py — pure logic, no I/O, no mocks needed.
"""
import pytest
from orchestrator.deduplicator import (
    _word_set,
    _jaccard,
    _is_active,
    deduplicate,
    pre_evaluate_dedup,
    SIMILARITY_THRESHOLD,
    PRE_EVAL_THRESHOLD,
)


# ── _word_set ──────────────────────────────────────────────────────────────────

class TestWordSet:
    def test_basic_tokenisation(self):
        assert _word_set("hello world") == {"hello", "world"}

    def test_lowercases(self):
        assert _word_set("Hello World") == {"hello", "world"}

    def test_strips_stop_words(self):
        result = _word_set("the value is in the list")
        assert "the" not in result
        assert "is" not in result
        assert "in" not in result
        assert "value" in result
        assert "list" in result

    def test_ignores_punctuation(self):
        assert _word_set("foo, bar! baz.") == {"foo", "bar", "baz"}

    def test_empty_string(self):
        assert _word_set("") == set()

    def test_only_stop_words(self):
        assert _word_set("the a an in") == set()


# ── _jaccard ───────────────────────────────────────────────────────────────────

class TestJaccard:
    def test_identical_sets(self):
        s = {"a", "b", "c"}
        assert _jaccard(s, s) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        # intersection = {b}, union = {a, b, c} → 1/3
        result = _jaccard({"a", "b"}, {"b", "c"})
        assert abs(result - 1 / 3) < 1e-9

    def test_empty_set_a(self):
        assert _jaccard(set(), {"a", "b"}) == 0.0

    def test_empty_set_b(self):
        assert _jaccard({"a", "b"}, set()) == 0.0

    def test_both_empty(self):
        assert _jaccard(set(), set()) == 0.0


# ── _is_active ─────────────────────────────────────────────────────────────────

class TestIsActive:
    def test_open_pr_is_active(self):
        assert _is_active({"pr_url": "https://github.com/x/y/pull/1", "pr_state": "open"})

    def test_pr_state_none_treated_as_open(self):
        assert _is_active({"pr_url": "https://github.com/x/y/pull/1", "pr_state": None})

    def test_approved_no_pr_is_active(self):
        assert _is_active({"pr_url": None, "approved": True})

    def test_merged_pr_is_not_active(self):
        assert not _is_active({"pr_url": "https://github.com/x/y/pull/1", "pr_state": "merged"})

    def test_closed_pr_is_not_active(self):
        assert not _is_active({"pr_url": "https://github.com/x/y/pull/1", "pr_state": "closed"})

    def test_no_pr_no_approval_is_not_active(self):
        assert not _is_active({"pr_url": None, "approved": False})

    def test_empty_dict_is_not_active(self):
        assert not _is_active({})


# ── deduplicate ────────────────────────────────────────────────────────────────

def _make_finding(file="app.py", category="security", description="hardcoded secret in config"):
    return {"file": file, "category": category, "description": description}


def _make_past(file="app.py", category="security", description="hardcoded secret in config",
               pr_url="https://github.com/x/y/pull/1", pr_state="open"):
    return {"file": file, "category": category, "description": description,
            "pr_url": pr_url, "pr_state": pr_state}


class TestDeduplicate:
    def test_empty_past_keeps_all(self):
        fresh = [_make_finding(), _make_finding(file="b.py")]
        kept, suppressed = deduplicate(fresh, [])
        assert len(kept) == 2
        assert suppressed == 0

    def test_suppresses_similar_active_finding(self):
        fresh = [_make_finding(description="hardcoded secret value in config file")]
        past = [_make_past(description="hardcoded secret in config")]
        kept, suppressed = deduplicate(fresh, past)
        assert suppressed == 1
        assert len(kept) == 0

    def test_keeps_finding_when_past_pr_merged(self):
        fresh = [_make_finding()]
        past = [_make_past(pr_state="merged")]
        kept, suppressed = deduplicate(fresh, past)
        assert suppressed == 0
        assert len(kept) == 1

    def test_keeps_finding_below_threshold(self):
        fresh = [_make_finding(description="completely different unrelated bug in authentication")]
        past = [_make_past(description="hardcoded secret in config")]
        kept, suppressed = deduplicate(fresh, past)
        assert suppressed == 0
        assert len(kept) == 1

    def test_keeps_finding_different_category(self):
        fresh = [_make_finding(category="performance")]
        past = [_make_past(category="security")]
        kept, suppressed = deduplicate(fresh, past)
        assert len(kept) == 1
        assert suppressed == 0

    def test_keeps_finding_different_file(self):
        fresh = [_make_finding(file="other.py")]
        past = [_make_past(file="app.py")]
        kept, suppressed = deduplicate(fresh, past)
        assert len(kept) == 1
        assert suppressed == 0


# ── pre_evaluate_dedup ─────────────────────────────────────────────────────────

class TestPreEvaluateDedup:
    def _finding(self, agent, file, category, description, severity="medium"):
        return {"agent": agent, "file": file, "category": category,
                "description": description, "severity": severity}

    def test_merges_cross_agent_duplicates(self):
        f1 = self._finding("infra_agent", "docker-compose.yml", "security",
                           "exposed mysql port 3306 to host network")
        f2 = self._finding("security_agent", "docker-compose.yml", "security",
                           "mysql port 3306 exposed to host network")
        kept, dropped = pre_evaluate_dedup([f1, f2])
        assert len(kept) == 1
        assert dropped == 1

    def test_keeps_distinct_findings(self):
        f1 = self._finding("infra_agent", "docker-compose.yml", "security",
                           "mysql port exposed to host network insecurely")
        f2 = self._finding("security_agent", "docker-compose.yml", "security",
                           "missing authentication header in nginx proxy config")
        kept, dropped = pre_evaluate_dedup([f1, f2])
        assert len(kept) == 2
        assert dropped == 0

    def test_picks_higher_severity_on_merge(self):
        low = self._finding("ml_agent", "app.py", "security",
                            "unvalidated input passed to shell command", severity="low")
        critical = self._finding("security_agent", "app.py", "security",
                                 "unvalidated user input passed to shell command", severity="critical")
        kept, dropped = pre_evaluate_dedup([low, critical])
        assert len(kept) == 1
        assert kept[0]["severity"] == "critical"

    def test_merges_agent_names(self):
        f1 = self._finding("infra_agent", "docker-compose.yml", "security",
                           "mysql port 3306 exposed to external host network")
        f2 = self._finding("security_agent", "docker-compose.yml", "security",
                           "mysql port 3306 exposed on host network interface")
        kept, _ = pre_evaluate_dedup([f1, f2])
        assert len(kept) == 1
        assert "infra_agent" in kept[0]["agent"]
        assert "security_agent" in kept[0]["agent"]

    def test_different_files_not_merged(self):
        f1 = self._finding("infra_agent", "docker-compose.yml", "security",
                           "mysql port exposed to host network")
        f2 = self._finding("security_agent", "kubernetes/database.yml", "security",
                           "mysql port exposed to host network")
        kept, dropped = pre_evaluate_dedup([f1, f2])
        assert len(kept) == 2
        assert dropped == 0

    def test_empty_input(self):
        kept, dropped = pre_evaluate_dedup([])
        assert kept == []
        assert dropped == 0
