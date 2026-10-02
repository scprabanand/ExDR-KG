import pytest
pytest.importorskip("pandas")
from exdrkg_rerun.reporting import language_audit as la


def test_glued_and_duplicate_detection():
    paras = ["The methodThe model works well.", "When μ = 0recovers the baseline we stop.", "This is a long repeated sentence about the graph encoder design. This is a long repeated sentence about the graph encoder design."]
    M = la.missing_spaces(paras, None)
    assert M.text.str.contains("methodThe").any() and M.text.str.contains("0recovers").any()
    assert len(la.duplicates(paras)) >= 1


def test_acronym_status():
    paras = ["We use a large language model (LLM) here.", "The LLM is strong.", "The large language model (LLM) is defined again.", "KG methods use a KG."]
    A = la.acronyms(paras).set_index("acronym")
    assert "DEFINED_TWICE" in A.loc["LLM", "status"] and "NEVER_DEFINED" in A.loc["KG", "status"]
