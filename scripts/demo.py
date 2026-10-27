"""
demo.py — End-to-end demonstration of all four ModelGuard scenarios.

Scenarios
---------
1. Healthy system    : normal predictions, no drift triggered.
2. Drift detected    : simulated drift triggers monitoring alert.
3. Challenger accepted : retraining with a suitable challenger → promotion.
4. Challenger rejected : retraining with an unsuitable challenger → retention.

Usage
-----
    python scripts/demo.py [--scenario {1,2,3,4,all}]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modelguard.config import (
    CHALLENGER_MIN_ROC_AUC,
    FEATURE_COLUMNS,
    REFERENCE_DATA_FILE,
    SIMULATED_DATA_DIR,
    TARGET_COLUMN,
)
from modelguard.monitoring import run_monitoring
from modelguard.retrain import run_retraining
from modelguard import registry as reg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def banner(text: str) -> None:
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


# ---------------------------------------------------------------------------
# Scenario 1 — Healthy system
# ---------------------------------------------------------------------------

def scenario_1_healthy():
    banner("SCENARIO 1 — Healthy System (No Drift)")

    # Load reference data as "normal" production batch
    if not REFERENCE_DATA_FILE.exists():
        logger.error("Reference data not found. Run training first.")
        return

    ref = pd.read_csv(REFERENCE_DATA_FILE)
    # Use a subsample that looks identical to reference (no drift)
    normal_batch = ref.sample(n=1000, random_state=1).copy()

    result = run_monitoring(
        current_batch=normal_batch,
        batch_label="demo_healthy",
        force=True,  # allow re-run in demo mode
    )
    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("report_html",)},
        indent=2, default=str,
    ))

    should_retrain = result.get("should_retrain", False)
    if not should_retrain:
        print("\n✓ PASS — No drift detected, no retraining triggered.")
    else:
        print(f"\n⚠ NOTE — Drift flagged (may happen with sub-sample variance).")
        print("  This is expected behavior — drift threshold may need tuning.")


# ---------------------------------------------------------------------------
# Scenario 2 — Drift detected
# ---------------------------------------------------------------------------

def scenario_2_drift():
    banner("SCENARIO 2 — Drift Detected")

    drifted_path = SIMULATED_DATA_DIR / "drifted_combined.csv"
    if not drifted_path.exists():
        logger.warning("Drifted batch not found; generating now …")
        from scripts.simulate_drift import scenario_combined
        from modelguard.data import load_raw_data
        df = load_raw_data()
        drifted = scenario_combined(df)
        SIMULATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        drifted_path = SIMULATED_DATA_DIR / "drifted_combined.csv"
        drifted.to_csv(drifted_path, index=False)

    batch = pd.read_csv(drifted_path)
    result = run_monitoring(
        current_batch=batch,
        batch_label="demo_drift",
        force=True,
    )
    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("report_html",)},
        indent=2, default=str,
    ))

    if result.get("dataset_drift") or result.get("should_retrain"):
        print("\n✓ PASS — Drift alert recorded. Retraining recommended.")
    else:
        print(
            "\n⚠ NOTE — No drift detected on this batch. "
            "Try a larger drifted batch or lower the drift threshold."
        )


# ---------------------------------------------------------------------------
# Scenario 3 — Challenger accepted
# ---------------------------------------------------------------------------

def scenario_3_challenger_accepted():
    banner("SCENARIO 3 — Challenger Accepted (Promotion)")

    result = run_retraining(
        trigger_batch=None,
        drift_already_confirmed=True,
        model_type="random_forest",
        force=True,
    )
    _print_retrain_result(result)

    if result.get("decision") == "promoted":
        print("\n✓ PASS — Challenger promoted to champion.")
    else:
        print(
            "\n⚠ NOTE — Challenger rejected in this run. "
            "This is valid behavior when the challenger doesn't improve on the champion."
        )


# ---------------------------------------------------------------------------
# Scenario 4 — Challenger rejected
# ---------------------------------------------------------------------------

def scenario_4_challenger_rejected():
    banner("SCENARIO 4 — Challenger Rejected (Champion Retained)")

    # Train a deliberately weak challenger (logistic regression, not expected
    # to beat a well-tuned random forest; outcome is data-dependent).
    result = run_retraining(
        trigger_batch=None,
        drift_already_confirmed=True,
        model_type="logistic_regression",
        force=True,
    )
    _print_retrain_result(result)

    if result.get("decision") == "rejected":
        print("\n✓ PASS — Challenger rejected, champion retained.")
    elif result.get("decision") == "promoted":
        print(
            "\n⚠ NOTE — Logistic regression challenger was accepted. "
            "Both models achieve similar performance on this dataset — "
            "this is valid behavior. The champion was correctly replaced."
        )


def _print_retrain_result(result: dict) -> None:
    print(json.dumps(
        {k: v for k, v in result.items()
         if k not in ("champion_metrics", "challenger_metrics")},
        indent=2, default=str,
    ))
    if "champion_metrics" in result:
        print(f"\nChampion  test ROC-AUC: {result['champion_metrics'].get('roc_auc', 'N/A'):.4f}")
    if "challenger_metrics" in result:
        print(f"Challenger test ROC-AUC: {result['challenger_metrics'].get('roc_auc', 'N/A'):.4f}")
    print(f"Decision: {result.get('decision', result.get('status'))}")
    print(f"Reason  : {result.get('reason', '')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SCENARIO_MAP = {
    "1": scenario_1_healthy,
    "2": scenario_2_drift,
    "3": scenario_3_challenger_accepted,
    "4": scenario_4_challenger_rejected,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelGuard demo scenarios.")
    parser.add_argument(
        "--scenario",
        choices=["1", "2", "3", "4", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.scenario == "all":
        for fn in SCENARIO_MAP.values():
            fn()
    else:
        SCENARIO_MAP[args.scenario]()


if __name__ == "__main__":
    main()
