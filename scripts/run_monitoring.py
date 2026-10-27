"""
run_monitoring.py — Run drift monitoring on a batch file.

Usage
-----
    # Monitor a simulated drifted batch:
    python scripts/run_monitoring.py --batch data/simulated/drifted_combined.csv

    # Monitor with a custom reference:
    python scripts/run_monitoring.py --batch data/simulated/drifted_combined.csv \
        --reference data/reference/reference.csv

    # Force reprocessing of an already-seen batch:
    python scripts/run_monitoring.py --batch data/simulated/drifted_combined.csv --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modelguard.monitoring import run_monitoring

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run drift monitoring on a batch.")
    parser.add_argument("--batch", type=Path, required=True, help="Path to batch CSV.")
    parser.add_argument(
        "--reference", type=Path, default=None, help="Path to reference CSV."
    )
    parser.add_argument(
        "--label", default="batch", help="Human-readable batch label."
    )
    parser.add_argument(
        "--force", action="store_true", help="Force reprocessing of seen batch."
    )
    args = parser.parse_args()

    if not args.batch.exists():
        logger.error("Batch file not found: %s", args.batch)
        sys.exit(1)

    current_batch = pd.read_csv(args.batch)
    reference_data = pd.read_csv(args.reference) if args.reference else None

    result = run_monitoring(
        current_batch=current_batch,
        reference_data=reference_data,
        batch_label=args.label or args.batch.stem,
        force=args.force,
    )

    logger.info("Monitoring result:")
    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("report_html",)},
        indent=2, default=str
    ))

    if result.get("should_retrain"):
        logger.warning(
            "⚠  Drift detected — retraining is recommended. "
            "Run: python scripts/run_retraining.py --batch %s",
            args.batch,
        )
    else:
        logger.info("✓ No retraining needed.")


if __name__ == "__main__":
    main()
