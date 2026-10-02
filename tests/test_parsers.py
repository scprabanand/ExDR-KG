import pytest


def test_parse_path_both_formats():
    pytest.importorskip("matplotlib")
    from exdrkg_rerun.reporting.figures import parse_path
    n, r = parse_path("Thalidomide -[binds]-> CRBN (gene) -[regulates]-> IKZF1 (gene)")
    assert n == ["Thalidomide", "CRBN", "IKZF1"] and r == ["binds", "regulates"]
    n, r = parse_path("A -> B -> C"); assert n == ["A", "B", "C"] and len(r) == 2


def test_parse_prob():
    pytest.importorskip("openai")
    from exdrkg_rerun.baselines.llm_api import parse_prob
    assert parse_prob(0.7) == 0.7 and parse_prob("85%") == 0.85 and parse_prob("85") == 0.85 and parse_prob(1.7) == 1.0 and parse_prob(None) is None and parse_prob(True) is None
