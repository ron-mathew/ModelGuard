"""Show deterministic acceptance and rejection policy examples.

These are synthetic metric inputs for the pure comparison policy. They do not
train models, create MLflow runs, or change the registered champion.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modelguard.evaluate import compare_models


def main() -> None:
    champion = {"roc_auc": 0.80}
    cases = {
        "accepted": {"roc_auc": 0.83},
        "rejected": {"roc_auc": 0.72},
    }

    for expected, challenger in cases.items():
        accepted, reason = compare_models(
            champion_metrics=champion,
            challenger_metrics=challenger,
            improvement_threshold=0.01,
            min_roc_auc=0.65,
        )
        decision = "accepted" if accepted else "rejected"
        if decision != expected:
            raise RuntimeError(
                f"Expected {expected}, got {decision}: {reason}"
            )
        printable_reason = reason.replace("Δ", "delta")
        print(f"{expected.upper()}: {printable_reason}")

    print("Policy examples only: synthetic metrics; no model or registry mutation.")


if __name__ == "__main__":
    main()
