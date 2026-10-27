"""
prepare_data.py — Validate and prepare the dataset, generate reference data.

Usage
-----
    python scripts/prepare_data.py
"""

import logging
import sys
from pathlib import Path

# Add src/ to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modelguard.config import RAW_DATA_FILE
from modelguard.data import (
    load_raw_data,
    prepare_features,
    save_manifest,
    save_reference_data,
    split_data,
    validate_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== ModelGuard — Data Preparation ===")

    # Load and validate
    df = load_raw_data()
    validate_data(df)

    # Split
    X, y = prepare_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Save manifest
    save_manifest(df, {"train": len(X_train), "val": len(X_val), "test": len(X_test)})

    # Save reference data for monitoring
    save_reference_data(X_train, y_train)

    logger.info("Data preparation complete.")
    logger.info("  Train : %d rows", len(X_train))
    logger.info("  Val   : %d rows", len(X_val))
    logger.info("  Test  : %d rows", len(X_test))


if __name__ == "__main__":
    main()
