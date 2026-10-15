"""Learner scaffold for the eligible-price recommendation workflow.

Complete the bounded TODOs in this file. The validation suite intentionally fails
until the regression, diagnostics, retrieval, and serving behavior is implemented.
"""

from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


FEATURE_COLUMNS = [
    "income",
    "credit_score",
    "employment_type",
    "loan_tenure_months",
    "down_payment",
]
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


class EligibilityRecommendationWorkflow:
    """Shape of the workflow learners must complete."""

    feature_columns = FEATURE_COLUMNS
    target_column = TARGET_COLUMN
    text_fields = TEXT_FIELDS

    def __init__(self) -> None:
        self.validation_report: dict[str, Any] = {}

    def predict_range(self, profile: dict[str, Any]) -> dict[str, float]:
        # TODO: transform the five profile fields with the fitted pipeline,
        # predict the eligible amount, and construct the exact 0.85/1.15 bounds.
        raise NotImplementedError("Implement prediction and non-negative price bounds")

    def recommend(
        self, profile: dict[str, Any], query: str, top_k: int
    ) -> dict[str, Any]:
        # TODO: hard-filter listings by price before fitting/applying TF-IDF,
        # rank by cosine similarity, ground each reason, and resolve ties by ID.
        raise NotImplementedError("Implement bounded and grounded retrieval")


def train_workflow(
    history: pd.DataFrame, listings: pd.DataFrame
) -> EligibilityRecommendationWorkflow:
    """Fit and return the workflow using the supplied in-memory datasets."""
    # TODO: validate required columns; train on rows 0:18 using only
    # FEATURE_COLUMNS; compute holdout and segmented diagnostics; and retain a
    # safe copy of the listing collection for retrieval.
    raise NotImplementedError("Implement the regression and validation workflow")


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

    # TODO: convert the request to the five-field profile, invoke recommend,
    # and preserve the required summary and educational disclaimer contract.
    raise NotImplementedError("Implement the FastAPI recommendation response")