import numpy as np
import pandas as pd
import pytest

from src.monitoring import feature_drift, prediction_drift, psi, psi_status, target_drift

RNG = np.random.default_rng(0)
N = 50_000


def test_psi_identical_distribution_is_near_zero():
    a, b = RNG.normal(10, 2, N), RNG.normal(10, 2, N)
    assert psi(a, b) < 0.01
    assert psi(a, a) == pytest.approx(0.0, abs=1e-12)


def test_psi_flags_mean_and_scale_shifts():
    ref = RNG.normal(10, 2, N)
    assert psi(ref, RNG.normal(11.5, 2, N)) > 0.2      # +0.75 sd
    assert psi(ref, RNG.normal(10, 4, N)) > 0.2        # doubled spread


def test_psi_zero_inflated_and_missing_values():
    ref = np.where(RNG.random(N) < 0.6, 0, RNG.poisson(3, N)).astype(float)   # 60% zeros, many ties
    same = np.where(RNG.random(N) < 0.6, 0, RNG.poisson(3, N)).astype(float)
    sparser = np.where(RNG.random(N) < 0.9, 0, RNG.poisson(3, N)).astype(float)
    assert psi(ref, same) < 0.01
    assert psi(ref, sparser) > 0.2
    with_nan = same.copy()
    with_nan[: N // 2] = np.nan                                                # half missing -> own bin
    assert psi(ref, with_nan) > 0.2


def test_psi_status_thresholds():
    assert [psi_status(x) for x in (0.05, 0.15, 0.25, float("nan"))] == ["ok", "watch", "flagged", "n/a"]


def test_feature_drift_flags_only_the_shifted_feature():
    ref = pd.DataFrame({"stable": RNG.normal(0, 1, N), "moving": RNG.normal(0, 1, N)})
    cur = pd.DataFrame({"stable": RNG.normal(0, 1, N), "moving": RNG.normal(1, 1, N)})
    last_year = pd.DataFrame({"stable": RNG.normal(0, 1, N), "moving": RNG.normal(1, 1, N)})
    out = feature_drift(ref, {"now": (cur, last_year)}, features=["stable", "moving"]).set_index("feature")
    assert out.loc["stable", "status"] == "ok" and out.loc["moving", "status"] == "flagged"
    assert out.loc["moving", "psi_vs_last_year"] < 0.01          # same shift a year ago -> seasonal, not drift
    assert out.loc["moving", "ks_stat"] > out.loc["stable", "ks_stat"]
    assert out.loc["moving", "cur_mean"] == pytest.approx(1.0, abs=0.05)


def test_feature_drift_without_last_year_window():
    df = pd.DataFrame({"x": RNG.normal(0, 1, 1000)})
    out = feature_drift(df, {"now": (df, None)}, features=["x"])
    assert out.loc[0, "psi_vs_last_year"] is None and out.loc[0, "status"] == "ok"


def test_target_drift_hand_computed():
    ref = pd.DataFrame({"cat_id": ["A", "A", "B", "B"], "sold": [0.0, 2.0, 0.0, 0.0]})
    cur = pd.DataFrame({"cat_id": ["A", "A", "B", "B"], "sold": [3.0, 3.0, 0.0, 2.0]})
    out = target_drift(ref, cur).set_index("category")
    assert out.loc["A", "ref_mean_daily"] == 1.0 and out.loc["A", "cur_mean_daily"] == 3.0
    assert out.loc["A", "change_pct"] == pytest.approx(200.0)
    assert out.loc["A", "zero_share_change_pp"] == pytest.approx(-50.0)       # 50% -> 0%
    assert out.loc["Total", "ref_mean_daily"] == 0.5 and out.loc["Total", "cur_mean_daily"] == 2.0


def test_prediction_drift_flags_large_gaps_only():
    fc = pd.DataFrame({"cat_id": ["A", "A", "B", "B"], "forecast": [1.1, 1.1, 2.0, 2.0]})
    act = pd.DataFrame({"cat_id": ["A", "A", "B", "B"], "sales": [1.0, 1.0, 1.0, 1.0]})
    out = prediction_drift(fc, act).set_index("category")
    assert out.loc["A", "gap_pct"] == pytest.approx(10.0) and out.loc["A", "status"] == "ok"
    assert out.loc["B", "gap_pct"] == pytest.approx(100.0) and out.loc["B", "status"] == "flagged"
    assert out.loc["Total", "gap_pct"] == pytest.approx(55.0)
