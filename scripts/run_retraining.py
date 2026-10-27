"""
run_retraining.py — Trigger the conditional retraining workflow.

Usage
-----
    # Retrain using reference data (no trigger batch):
    python scripts/run_retraining.py

    # Retrain triggered by a specific drifted batch:
    python scripts/run_retraining.py --batch data/simulated/drifted_combined.csv

    # Force retrain (bypass duplicate guard):
    python scripts/run_retraining.py --force

    # Choose model type:
    python scripts/run_retraining.py --model random_forest
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modelguard.retrain import run_retraining

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conditional retraining.")
    parser.add_argument(
        "--batch", type=Path, default=None,
        help="Trigger batch CSV (optional)."
    )
    parser.add_argument(
        "--model", choices=["random_forest", "logistic_regression"],
        default="random_forest", help="Challenger model type."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass duplicate-batch guard."
    )
    parser.add_argument(
        "--confirmed-drift", action="store_true",
        help="Skip drift detection because this batch was already confirmed.",
    )
    args = parser.parse_args()

    trigger_batch = None
    if args.batch:
        if not args.batch.exists():
            logger.error("Batch file not found: %s", args.batch)
            sys.exit(1)
        trigger_batch = pd.read_csv(args.batch)
        logger.info("Trigger batch loaded: %d rows.", len(trigger_batch))

    result = run_retraining(
        trigger_batch=trigger_batch,
        drift_already_confirmed=args.confirmed_drift,
        model_type=args.model,
        force=args.force,
    )

    print("\n=== Retraining Result ===")
    print(json.dumps(
        {k: v for k, v in result.items()
         if k not in ("champion_metrics", "challenger_metrics")},
        indent=2, default=str
    ))
    if "champion_metrics" in result:
        print(f"\nChampion test ROC-AUC : {result['champion_metrics'].get('roc_auc', 'N/A'):.4f}")
    if "challenger_metrics" in result:
        print(f"Challenger test ROC-AUC: {result['challenger_metrics'].get('roc_auc', 'N/A'):.4f}")
    print(f"Decision: {result.get('decision', result.get('status'))}")
    print(f"Reason  : {result.get('reason', '')}")


if __name__ == "__main__":
    main()
