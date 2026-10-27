"""
Preprocessing pipeline construction.

Design decisions
----------------
- Pipeline is fitted ONLY on training data and applied to val/test.
- Numerical: median imputation → StandardScaler.
- Categorical: most-frequent imputation → OneHotEncoder (handle_unknown='ignore').
- ColumnTransformer assembles both sub-pipelines and passes through nothing else.
- The final artifact is a full sklearn Pipeline (preprocessor + estimator).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modelguard.config import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_IMPUTE_STRATEGY,
    HANDLE_UNKNOWN_CATEGORIES,
    NUMERICAL_FEATURES,
    NUMERICAL_IMPUTE_STRATEGY,
)


def build_preprocessor() -> ColumnTransformer:
    """
    Return a ColumnTransformer that handles numerical and categorical features.

    Numerical pipeline  : SimpleImputer(median) → StandardScaler
    Categorical pipeline: SimpleImputer(most_frequent) → OneHotEncoder
    """
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy=NUMERICAL_IMPUTE_STRATEGY),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy=CATEGORICAL_IMPUTE_STRATEGY),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown=HANDLE_UNKNOWN_CATEGORIES,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # drop any unexpected columns
    )
    return preprocessor


def build_full_pipeline(estimator) -> Pipeline:
    """
    Wrap the preprocessor + estimator into a single sklearn Pipeline.

    Parameters
    ----------
    estimator : sklearn-compatible classifier

    Returns
    -------
    Pipeline
        A pipeline whose .fit(X, y) trains end-to-end and
        .predict_proba(X) runs inference end-to-end.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )
