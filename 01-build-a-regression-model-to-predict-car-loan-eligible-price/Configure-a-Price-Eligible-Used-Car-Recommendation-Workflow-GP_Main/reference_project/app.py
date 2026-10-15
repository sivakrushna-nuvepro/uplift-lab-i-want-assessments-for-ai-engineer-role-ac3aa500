"""Eligibility regression and grounded used-car recommendation workflow."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS = [
    "income",
    "credit_score",
    "employment_type",
    "loan_tenure_months",
    "down_payment",
]
NUMERIC_COLUMNS = [
    "income",
    "credit_score",
    "loan_tenure_months",
    "down_payment",
]
CATEGORICAL_COLUMNS = ["employment_type"]
TARGET_COLUMN = "eligible_amount"
TEXT_FIELDS = ["service_history", "condition_notes", "other_details"]
DISCLAIMER = "Educational shopping estimate only; not a credit approval or financial advice."


class RecommendationRequest(BaseModel):
    income: float
    credit_score: int
    employment_type: Optional[str]
    loan_tenure_months: int
    down_payment: float
    query: str
    top_k: int = Field(default=5, ge=1)


def _profile_frame(profile: dict[str, Any]) -> pd.DataFrame:
    missing = [name for name in FEATURE_COLUMNS if name not in profile]
    if missing:
        raise ValueError(f"Missing profile fields: {', '.join(missing)}")
    frame = pd.DataFrame([{name: profile[name] for name in FEATURE_COLUMNS}])
    frame["employment_type"] = frame["employment_type"].replace({None: np.nan})
    return frame


def _build_model() -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_COLUMNS),
            ("employment", categorical, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("regression", LinearRegression()),
        ]
    )


def _segment_statistics(
    labels: pd.Series,
    actual: np.ndarray,
    predicted: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    label_values = labels.astype("string").fillna("Missing").to_numpy()
    for label in sorted(set(str(value) for value in label_values)):
        mask = np.asarray([str(value) == label for value in label_values])
        segment_actual = actual[mask]
        segment_predicted = predicted[mask]
        covered = (
            (segment_actual >= lower[mask]) & (segment_actual <= upper[mask])
        )
        output[label] = {
            "count": int(mask.sum()),
            "mae": float(mean_absolute_error(segment_actual, segment_predicted)),
            "coverage": float(np.mean(covered)),
        }
    return output


class EligibilityRecommendationWorkflow:
    feature_columns = FEATURE_COLUMNS
    target_column = TARGET_COLUMN
    text_fields = TEXT_FIELDS

    def __init__(
        self,
        model: Pipeline,
        listings: pd.DataFrame,
        validation_report: dict[str, Any],
    ) -> None:
        self.model = model
        self.listings = listings.copy(deep=True)
        self.validation_report = validation_report

    def predict_range(self, profile: dict[str, Any]) -> dict[str, float]:
        raw_prediction = float(self.model.predict(_profile_frame(profile))[0])
        predicted_price = max(0.0, raw_prediction)
        lower_bound = max(0.0, predicted_price * 0.85)
        upper_bound = predicted_price * 1.15
        return {
            "predicted_price": predicted_price,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

    def recommend(
        self, profile: dict[str, Any], query: str, top_k: int
    ) -> dict[str, Any]:
        bounds = self.predict_range(profile)
        lower = bounds["lower_bound"]
        upper = bounds["upper_bound"]

        prices = pd.to_numeric(self.listings["price"], errors="coerce")
        eligible = self.listings.loc[prices.between(lower, upper)].copy()
        eligible["price"] = pd.to_numeric(eligible["price"], errors="coerce")
        eligible_count = int(len(eligible))

        recommendations: list[dict[str, Any]] = []
        if eligible_count:
            retrieval_text = eligible[TEXT_FIELDS].fillna("").astype(str).agg(" ".join, axis=1)
            corpus = retrieval_text.tolist() + [str(query or "")]
            vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
            vectors = vectorizer.fit_transform(corpus)
            scores = cosine_similarity(vectors[:-1], vectors[-1]).reshape(-1)
            eligible["_similarity"] = np.clip(scores, 0.0, 1.0)
            eligible["_listing_sort"] = eligible["listing_id"].astype(str)
            eligible = eligible.sort_values(
                by=["_similarity", "_listing_sort"],
                ascending=[False, True],
                kind="mergesort",
            )

            query_terms = set(str(query or "").lower().split())
            limit = max(0, int(top_k))
            for rank, (_, row) in enumerate(eligible.head(limit).iterrows(), start=1):
                fields = [str(row[field]) for field in TEXT_FIELDS]
                excerpt = max(
                    fields,
                    key=lambda value: (
                        len(query_terms.intersection(value.lower().split())),
                        -fields.index(value),
                    ),
                )
                recommendations.append(
                    {
                        "rank": rank,
                        "listing_id": str(row["listing_id"]),
                        "price": float(row["price"]),
                        "similarity_score": float(row["_similarity"]),
                        "evidence_excerpt": excerpt,
                        "reason": f"Ranked using listing evidence: {excerpt}",
                    }
                )

        if recommendations:
            identifiers = ", ".join(item["listing_id"] for item in recommendations)
            summary = (
                f"Eligible in-range listings ranked from the supplied service and "
                f"condition evidence: {identifiers}."
            )
        else:
            summary = "No supplied used-car listings fall within the estimated price range."

        return {
            **bounds,
            "eligible_listing_count": eligible_count,
            "recommendations": recommendations,
            "summary": summary,
            "disclaimer": DISCLAIMER,
        }


def train_workflow(
    history: pd.DataFrame, listings: pd.DataFrame
) -> EligibilityRecommendationWorkflow:
    required_history = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_history = sorted(required_history.difference(history.columns))
    if missing_history:
        raise ValueError(f"Borrower history is missing columns: {missing_history}")

    required_listings = {"listing_id", "price", *TEXT_FIELDS}
    missing_listings = sorted(required_listings.difference(listings.columns))
    if missing_listings:
        raise ValueError(f"Listings are missing columns: {missing_listings}")
    if len(history) < 24:
        raise ValueError("Borrower history must contain 18 training and 6 holdout rows")

    training = history.iloc[:18].copy()
    holdout = history.iloc[18:24].copy()
    model = _build_model()
    model.fit(training[FEATURE_COLUMNS], training[TARGET_COLUMN])

    actual = holdout[TARGET_COLUMN].to_numpy(dtype=float)
    predicted = np.maximum(
        0.0, model.predict(holdout[FEATURE_COLUMNS]).astype(float)
    )
    lower = np.maximum(0.0, predicted * 0.85)
    upper = predicted * 1.15
    baseline_value = float(training[TARGET_COLUMN].median())
    baseline = np.full(shape=len(holdout), fill_value=baseline_value, dtype=float)
    covered = (actual >= lower) & (actual <= upper)

    income_band = pd.cut(
        holdout["income"],
        bins=[-np.inf, 30000, 38000, np.inf],
        labels=["low", "middle", "high"],
    )
    credit_band = pd.cut(
        holdout["credit_score"],
        bins=[-np.inf, 640, 680, np.inf],
        labels=["low", "middle", "high"],
    )
    employment = holdout["employment_type"].astype("string").fillna("Missing")

    validation_report = {
        "holdout_count": int(len(holdout)),
        "model_mae": float(mean_absolute_error(actual, predicted)),
        "baseline_mae": float(mean_absolute_error(actual, baseline)),
        "range_coverage": float(np.mean(covered)),
        "segments": {
            "income_band": _segment_statistics(
                pd.Series(income_band, index=holdout.index),
                actual,
                predicted,
                lower,
                upper,
            ),
            "credit_score_band": _segment_statistics(
                pd.Series(credit_band, index=holdout.index),
                actual,
                predicted,
                lower,
                upper,
            ),
            "employment_type": _segment_statistics(
                employment,
                actual,
                predicted,
                lower,
                upper,
            ),
        },
    }
    return EligibilityRecommendationWorkflow(model, listings, validation_report)


app = FastAPI(title="Eligible-Price Used-Car Recommendations")
_serving_workflow: Optional[EligibilityRecommendationWorkflow] = None


def configure_workflow(
    history: pd.DataFrame, listings: pd.DataFrame
) -> EligibilityRecommendationWorkflow:
    global _serving_workflow
    _serving_workflow = train_workflow(history, listings)
    return _serving_workflow


@app.post("/recommend")
def recommend_endpoint(request: RecommendationRequest) -> dict[str, Any]:
    if _serving_workflow is None:
        raise HTTPException(status_code=503, detail="Workflow has not been configured")
    payload = request.model_dump()
    query = payload.pop("query")
    top_k = payload.pop("top_k")
    return _serving_workflow.recommend(payload, query=query, top_k=top_k)