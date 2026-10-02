import numpy as np, pandas as pd
from exdrkg_rerun.common.metrics import compute_metrics, bootstrap_ci, delong_test, paired_bootstrap_p


def frame():
    rng = np.random.default_rng(0)
    rows = [(f"d{i % 5}", f"x{i}", int(i % 4 == 0)) for i in range(200)]
    df = pd.DataFrame(rows, columns=["drug_id", "disease_id", "label"])
    df["disease_id"], df["drug_id"] = df["drug_id"], df["disease_id"]
    return df, rng


def test_perfect_ranker_scores_one():
    df, _ = frame(); m = compute_metrics(df, df.label.values + np.linspace(0, 0.1, len(df)))
    assert m["AUROC"] == 1.0 and m["AUPRC"] == 1.0 and m["Hit@10"] == 1.0


def test_constant_score_is_chance_and_ties_share_rank():
    df, _ = frame(); m = compute_metrics(df, np.zeros(len(df)))
    assert abs(m["AUROC"] - 0.5) < 1e-9 and m["Hit@1"] < 0.5


def test_bootstrap_is_deterministic_and_brackets_estimate():
    df, rng = frame(); s = df.label.values + rng.normal(0, 0.8, len(df))
    a, b = bootstrap_ci(df, s, n=100), bootstrap_ci(df, s, n=100)
    assert a == b and a["AUROC"][0] <= compute_metrics(df, s)["AUROC"] <= a["AUROC"][1]


def test_delong_detects_better_model():
    df, rng = frame(); good = df.label.values + rng.normal(0, 0.3, len(df)); bad = rng.normal(0, 1, len(df))
    a1, a2, p = delong_test(df.label.values, good, bad)
    assert a1 > a2 and p < 0.05
    assert paired_bootstrap_p(df, good, bad, "AUROC", n=100) < 0.1
