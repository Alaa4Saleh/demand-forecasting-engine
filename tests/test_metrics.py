import numpy as np
import pandas as pd

from src.metrics import WRMSSEEvaluator, _scale, hierarchy_from_ids

IDS = pd.Index([
    "HOBBIES_1_001_CA_1_validation",
    "HOBBIES_1_002_CA_1_validation",
    "FOODS_3_001_TX_2_validation",
])


def _evaluator(horizon=2):
    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.poisson(3, size=(3, 20)).astype(float), index=IDS)
    dollars = pd.Series([10.0, 30.0, 60.0], index=IDS)
    return WRMSSEEvaluator(train, dollars, horizon=horizon)


def test_hierarchy_parsing():
    h = hierarchy_from_ids(IDS).iloc[2]
    assert (h.item_id, h.dept_id, h.cat_id, h.store_id, h.state_id) == \
        ("FOODS_3_001", "FOODS_3", "FOODS", "TX_2", "TX")


def test_scale_starts_at_first_nonzero():
    # leading zeros are ignored: diffs of [2, 4, 4] -> (4 + 0) / 2
    assert _scale(np.array([[0, 0, 2, 4, 4]], dtype=np.float32))[0] == 2.0


def test_perfect_forecast_scores_zero():
    ev = _evaluator()
    y = np.ones((3, 2))
    assert ev.score(y, y) == 0.0


def test_level_weights_sum_to_one_and_count():
    ev = _evaluator()
    assert len(ev.levels) == 12
    for _, _, w in ev.levels.values():
        assert np.isclose(w.sum(), 1.0)


def test_series_scores_sum_to_level_score():
    ev = _evaluator()
    y, p = np.zeros((3, 2)), np.full((3, 2), 2.0)
    for level in ("store", "item_store"):
        s = ev.series_scores(y, p, level)
        assert np.isclose(s["contribution"].sum(), ev.level_scores(y, p)[level])
    assert set(ev.series_scores(y, p, "store")["store_id"]) == {"CA_1", "TX_2"}


def test_bottom_level_matches_manual_rmsse():
    ev = _evaluator()
    y, p = np.zeros((3, 2)), np.ones((3, 2))
    _, scale, weight = ev.levels["item_store"]
    expected = (weight * np.sqrt(1.0 / scale)).sum()
    assert np.isclose(ev.level_scores(y, p)["item_store"], expected)
