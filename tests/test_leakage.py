"""TEST_PLAN.md section 1: leakage detection."""

import numpy as np
import pandas as pd

from level4_feature.feature_catalog import get_catalog
from level5_policy.leakage_detector import LeakageDetector, assembled_dates, joint_leaks
from tests.conftest import StubLLM, make_intent, synthetic_df


def test_clean_data_has_no_leak_flags(run):
    code, orch = run(synthetic_df())
    assert code == 0
    assert orch.leakage_dropped_columns == set()


def test_1_1_planted_target_copy_is_dropped(run):
    df = synthetic_df()
    df["final_review"] = df["target"]
    code, orch = run(df)
    assert code == 0
    assert "final_review" in orch.leakage_dropped_columns
    assert "final_review" not in orch.feature_dataframe.columns


def test_1_2_missingness_leak_is_dropped(run):
    # Like Titanic `body`: only recorded for one outcome
    df = synthetic_df()
    rng = np.random.default_rng(1)
    df["body"] = np.where((df["target"] == 0) & (rng.random(len(df)) < 0.3), rng.integers(1, 300, len(df)), np.nan)
    _, orch = run(df)
    assert "body" in orch.leakage_dropped_columns


def test_1_3_near_perfect_predictor_is_dropped(run):
    # Like Titanic `boat`: mostly present for survivors, with a little noise
    df = synthetic_df()
    rng = np.random.default_rng(2)
    flip = rng.random(len(df)) < 0.01
    df["boat"] = np.where((df["target"] == 1) ^ flip, rng.choice(list("ABCD"), len(df)), "none")
    _, orch = run(df)
    assert "boat" in orch.leakage_dropped_columns


def test_1_4_llm_suspected_leak_is_surfaced_not_dropped(run):
    llm = StubLLM(suspected_leaks=[{"column": "num_b", "reason": "recorded after the outcome"}])
    code, orch = run(synthetic_df(), llm=llm)
    assert code == 0
    assert "num_b" in orch.feature_dataframe.columns
    assert any("Suspected leak 'num_b'" in w for w in orch.governance_decision.warnings)


def test_1_4_bogus_suspected_leaks_are_ignored(run):
    llm = StubLLM(suspected_leaks=[{"column": "target", "reason": "x"}, {"column": "nope"}, "junk", {"reason": "x"}])
    _, orch = run(synthetic_df(), llm=llm)
    assert orch.suspected_leaks == []


def test_1_5_and_1_6_names_alone_never_flag(run):
    # Old detector flagged anything containing "y_" or the target name ("y" -> "day")
    df = synthetic_df().rename(columns={"target": "y", "num_a": "family_size", "num_b": "day", "cat_c": "monthly_charges"})
    llm = StubLLM(features=[{"name": "day_bin_y_label", "source_columns": ["day"], "transformation": "bin_uniform"}])
    code, orch = run(df, llm=llm, target="y")
    assert code == 0
    assert orch.leakage_dropped_columns == set()
    assert "day_bin_y_label" in orch.feature_dataframe.columns


def test_1_7_catalog_has_no_cross_row_transforms():
    # Temporal leakage is ruled out only because no transform reads other rows.
    # Adding lag/rolling/shift transforms requires a real temporal leak check first.
    names = {t.name for t in get_catalog().list_all()}
    assert not {n for n in names if any(k in n for k in ("lag", "lead", "shift", "rolling", "window", "diff"))}


def test_1_8_target_as_feature_input_is_rejected(run):
    llm = StubLLM(features=[{"name": "t_log", "source_columns": ["target"], "transformation": "log"}])
    _, orch = run(synthetic_df(), llm=llm)
    assert [r.name for r in orch.rejected_features] == ["t_log"]
    assert "t_log" not in orch.feature_dataframe.columns


def test_regression_target_copy_is_dropped(run):
    df = synthetic_df()
    df["price"] = df["num_a"] * 10 + np.random.default_rng(3).normal(size=len(df)) * 5
    df["price_copy"] = df["price"] * 2
    _, orch = run(df.drop(columns="target"), target="price", task="regression")
    assert orch.leakage_dropped_columns == {"price_copy"}


def test_low_precision_numeric_columns_do_not_crash():
    # Generated features can be float16 (seen on Bank Marketing); qcut rejects float16
    df = synthetic_df()
    df["half"] = df["num_b"].astype("float16")
    detector = LeakageDetector(make_intent("unused.csv"))
    assert detector.find_leaks({"feature_dataframe": df, "target_column": "target"}) == {}


def test_imputation_does_not_hide_a_missingness_leak(run):
    # Seen on Titanic: the LLM imputed `body`, erasing the signal before governance
    df = synthetic_df()
    rng = np.random.default_rng(1)
    df["body"] = np.where((df["target"] == 0) & (rng.random(len(df)) < 0.3), rng.integers(1, 300, len(df)), np.nan)
    llm = StubLLM(
        actions=[{"column": "body", "action": "impute", "method": "median"}],
        features=[{"name": "body_log", "source_columns": ["body"], "transformation": "log"}],
    )
    _, orch = run(df, llm=llm)
    assert {"body", "body_log"} <= orch.leakage_dropped_columns


def hotel_like_df(seed=3, junk=0.02):
    """Like Hotel Booking `reservation_status_date`: set before arrival if canceled, after it if not.

    Arrival is split into year / month name / day columns. Neither date predicts the target alone.
    """
    df = synthetic_df()
    rng = np.random.default_rng(seed)
    arrival = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 700, len(df)), unit="D")
    df["arrival_year"], df["arrival_month"], df["arrival_day_of_month"] = arrival.year, arrival.month_name(), arrival.day
    gap = np.where(df["target"] == 1, -rng.integers(1, 200, len(df)), rng.integers(0, 10, len(df)))
    status = (arrival + pd.to_timedelta(gap, unit="D")).strftime("%Y-%m-%d").to_series(index=df.index)
    df["status_date"] = status.mask(rng.random(len(df)) < junk, "not a date")
    return df


def xor_df(seed=4):
    """Neither `a` nor `b` predicts the target; together they determine it."""
    df = synthetic_df()
    rng = np.random.default_rng(seed)
    df["a"], df["b"] = rng.integers(0, 4, len(df)), rng.integers(0, 4, len(df))
    df["target"] = (df["a"] + df["b"]) % 2
    return df


def test_1_11_joint_leak_confirmed_by_llm_is_dropped(run):
    llm = StubLLM(
        suspected_leaks=[{"column": "status_date", "reason": "set at checkout"}],
        features=[{"name": "status_year", "source_columns": ["status_date"], "transformation": "extract_year"}],
    )
    _, orch = run(hotel_like_df(junk=0), llm=llm)  # L1 reads dates with junk as text
    assert {"status_date", "status_year"} <= orch.leakage_dropped_columns
    assert not {"arrival_year", "arrival_month", "arrival_day_of_month"} & orch.leakage_dropped_columns


def test_1_11_dates_with_unparseable_values_still_pair_up():
    df = hotel_like_df()
    pairs = joint_leaks(df.drop(columns="target"), df["target"], True)
    assert ("status_date", "arrival_year/month/day") in [p[:2] for p in pairs]


def test_1_11_unattributed_joint_leak_is_warned_not_dropped(run):
    code, orch = run(xor_df())
    assert code == 0
    assert {"a", "b"} <= set(orch.feature_dataframe.columns)
    assert any("'a'" in w and "'b'" in w and "together" in w for w in orch.governance_decision.warnings)


def test_1_11_single_column_leak_is_not_reported_as_a_pair():
    df = synthetic_df()
    df["copy"] = df["target"]
    pairs = joint_leaks(df.drop(columns="target"), df["target"], True)
    assert all("copy" not in pair[:2] for pair in pairs)


def test_1_11_clean_pairs_are_not_flagged():
    df = hotel_like_df()
    df["status_date"] = df["status_date"].sample(frac=1, random_state=0).values  # break the link
    assert joint_leaks(df.drop(columns="target"), df["target"], True) == []


def test_assembled_dates_parse_names_numbers_and_bad_days():
    df = pd.DataFrame({"a_year": [2020, 2020, 2021], "a_month": ["July", "2", "Feb"], "a_day_of_month": [4, 30, 1]})
    d = assembled_dates(df)["a_year/month/day"]
    assert d[0] == pd.Timestamp("2020-07-04") and pd.isna(d[1]) and d[2] == pd.Timestamp("2021-02-01")
