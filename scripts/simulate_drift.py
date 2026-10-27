"""
simulate_drift.py — Generate controlled drift scenarios from bank-full.csv.

The original dataset is NEVER modified.  Drifted data is written to
data/simulated/.

Drift scenarios implemented
----------------------------
1. age_shift    : shift age distribution by +10 years
2. job_shift    : over-represent 'retired' and 'student' categories
3. campaign_amp : multiply campaign contacts by 3 (heavy callers)
4. combined     : all of the above simultaneously

Usage
-----
    python scripts/simulate_drift.py [--scenario {age_shift,job_shift,campaign_amp,combined,all}]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modelguard.config import (
    FEATURE_COLUMNS,
    RAW_DATA_FILE,
    RANDOM_SEED,
    SIMULATED_DATA_DIR,
    TARGET_COLUMN,
)
from modelguard.data import load_raw_data, prepare_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

rng = np.random.default_rng(RANDOM_SEED)
N_SIMULATED = 2000  # number of rows per simulated batch


def _base_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Random sample of N rows with fresh unique indexes and target included."""
    # Sampling with replacement preserves duplicate source indexes. The
    # scenario mutators assign by index, so reset to unique indexes first.
    return (
        df.sample(n=N_SIMULATED, replace=True, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )


def scenario_age_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Shift all ages up by 10 years (capped at 95)."""
    sample = _base_sample(df)
    sample["age"] = (sample["age"] + 10).clip(upper=95)
    logger.info(
        "age_shift: mean age %.1f → %.1f",
        df["age"].mean(), sample["age"].mean(),
    )
    return sample


def scenario_job_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Over-represent retired and student job categories (80% of batch)."""
    sample = _base_sample(df)
    n_replace = int(len(sample) * 0.8)
    replace_idx = rng.choice(sample.index, size=n_replace, replace=False)
    sample.loc[replace_idx, "job"] = rng.choice(
        ["retired", "student"], size=n_replace
    )
    logger.info(
        "job_shift: retired+student fraction %.1f%%",
        (sample["job"].isin(["retired", "student"]).mean() * 100),
    )
    return sample


def scenario_campaign_amp(df: pd.DataFrame) -> pd.DataFrame:
    """Triple the number of campaign contacts."""
    sample = _base_sample(df)
    sample["campaign"] = (sample["campaign"] * 3).clip(upper=63)
    logger.info(
        "campaign_amp: mean campaign %.1f → %.1f",
        df["campaign"].mean(), sample["campaign"].mean(),
    )
    return sample


def scenario_combined(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all drift scenarios simultaneously."""
    sample = scenario_age_shift(df)
    sample["job"] = scenario_job_shift(df)["job"].values
    sample["campaign"] = scenario_campaign_amp(df)["campaign"].values
    return sample


SCENARIOS = {
    "age_shift": scenario_age_shift,
    "job_shift": scenario_job_shift,
    "campaign_amp": scenario_campaign_amp,
    "combined": scenario_combined,
}


def generate(scenario_name: str, df: pd.DataFrame) -> Path:
    fn = SCENARIOS[scenario_name]
    drifted = fn(df)
    SIMULATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SIMULATED_DATA_DIR / f"drifted_{scenario_name}.csv"
    drifted.to_csv(out_path, index=False)
    logger.info("Drifted batch saved: %s  (%d rows)", out_path, len(drifted))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate drift scenarios.")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="combined",
    )
    args = parser.parse_args()

    df = load_raw_data()

    if args.scenario == "all":
        for name in SCENARIOS:
            generate(name, df)
    else:
        generate(args.scenario, df)

    logger.info("Drift simulation complete.")


if __name__ == "__main__":
    main()
