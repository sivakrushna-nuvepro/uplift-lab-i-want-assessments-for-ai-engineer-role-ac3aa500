import asyncio
import importlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI


VALIDATION_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = VALIDATION_DIR.parent
PRACTICE_PROJECT = REPOSITORY_ROOT / "practice_project"
FIXTURES = VALIDATION_DIR / "fixtures"

if str(PRACTICE_PROJECT) not in sys.path:
    sys.path.insert(0, str(PRACTICE_PROJECT))

solution = importlib.import_module("app")

FEATURES = [
    "income",
    "credit_score",
    "employment_type",
    "loan_tenure_months",
    "down_payment",
]
LEAKAGE_FIELDS = [
    "purchased_vehicle_price",
    "approved_interest_rate",
    "final_financed_amount",
]
TEXT_FIELDS = ["service_history", "condition_notes", "other_details"]
DISCLAIMER = "Educational shopping estimate only; not a credit approval or financial advice."


def load_history():
    return pd.read_csv(FIXTURES / "borrower_history.csv")


def load_listings():
    return pd.read_csv(FIXTURES / "car_listings.csv")


def profile(employment_type="Salaried"):
    return {
        "income": 30000.0,
        "credit_score": 660,
        "employment_type": employment_type,
        "loan_tenure_months": 48,
        "down_payment": 4000.0,
    }


def as_mapping(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    raise AssertionError(f"Expected a mapping-compatible result, got {type(value)!r}")


@pytest.fixture
def workflow():
    return solution.train_workflow(load_history(), load_listings())


class TestRegressionWorkflow:
    def test_regression_uses_only_predecision_features(self):
        history = load_history()
        listings = load_listings()
        fitted = solution.train_workflow(history.copy(), listings.copy())

        assert list(fitted.feature_columns) == FEATURES
        assert fitted.target_column == "eligible_amount"
        assert all(field in history.columns for field in LEAKAGE_FIELDS)
        assert not set(LEAKAGE_FIELDS).intersection(fitted.feature_columns)

        changed = history.copy()
        changed["purchased_vehicle_price"] = np.arange(len(changed)) * 9_000_000 + 1
        changed["approved_interest_rate"] = np.arange(len(changed)) * 17 + 80
        changed["final_financed_amount"] = np.arange(len(changed)) * -7_000_000
        changed_fit = solution.train_workflow(changed, listings.copy())

        original_prediction = fitted.predict_range(profile())["predicted_price"]
        changed_prediction = changed_fit.predict_range(profile())["predicted_price"]
        assert changed_prediction == pytest.approx(original_prediction, abs=1e-8)

    def test_prediction_bounds_and_unknown_employment(self, workflow):
        known = workflow.predict_range(profile("Salaried"))
        unknown = workflow.predict_range(profile("GigWorker"))
        missing = workflow.predict_range(profile(None))

        for result in (known, unknown, missing):
            assert set(result) >= {"predicted_price", "lower_bound", "upper_bound"}
            assert np.isfinite(result["predicted_price"])
            assert 0 <= result["lower_bound"] <= result["predicted_price"] <= result["upper_bound"]
            assert result["lower_bound"] == pytest.approx(
                max(0.0, result["predicted_price"] * 0.85), abs=1e-6
            )
            assert result["upper_bound"] == pytest.approx(
                result["predicted_price"] * 1.15, abs=1e-6
            )

        assert known["predicted_price"] == pytest.approx(31000.0, abs=2500.0)


class TestModelValidation:
    def test_validation_beats_baseline_and_reports_segments(self, workflow):
        report = workflow.validation_report

        assert report["holdout_count"] == 6
        assert 0 <= report["model_mae"] < report["baseline_mae"]
        assert report["model_mae"] < 1800.0
        assert 0.80 <= report["range_coverage"] <= 1.0
        assert set(report["segments"]) == {
            "income_band",
            "credit_score_band",
            "employment_type",
        }

        for dimension in ("income_band", "credit_score_band", "employment_type"):
            segment_rows = report["segments"][dimension]
            assert len(segment_rows) >= 2
            assert sum(row["count"] for row in segment_rows.values()) == 6
            for row in segment_rows.values():
                assert row["count"] >= 1
                assert row["mae"] >= 0
                assert 0 <= row["coverage"] <= 1


class TestRetrievalAndApi:
    def test_retrieval_filters_before_ranking_and_is_grounded(self, workflow):
        query = "full dealer service history excellent condition no accidents"
        result = workflow.recommend(profile(), query=query, top_k=4)
        recommendations = result["recommendations"]
        source = load_listings().set_index("listing_id")

        assert result["eligible_listing_count"] == 6
        assert len(recommendations) == 4
        assert recommendations[0]["listing_id"] == "CAR-101"
        assert [item["rank"] for item in recommendations] == [1, 2, 3, 4]

        returned_ids = {item["listing_id"] for item in recommendations}
        assert "CAR-104" in source.index
        assert "CAR-105" in source.index
        assert "CAR-104" not in returned_ids
        assert "CAR-105" not in returned_ids

        for item in recommendations:
            listing = source.loc[item["listing_id"]]
            assert result["lower_bound"] <= item["price"] <= result["upper_bound"]
            assert item["price"] == pytest.approx(float(listing["price"]))
            valid_excerpts = {str(listing[field]) for field in TEXT_FIELDS}
            assert item["evidence_excerpt"] in valid_excerpts
            assert item["evidence_excerpt"] in item["reason"]
            assert 0.0 <= item["similarity_score"] <= 1.0

    def test_retrieval_ties_are_deterministic(self, workflow):
        query = "carefully maintained clean interior recent inspection"
        first = workflow.recommend(profile(), query=query, top_k=10)
        second = workflow.recommend(profile(), query=query, top_k=10)
        first_ids = [item["listing_id"] for item in first["recommendations"]]
        second_ids = [item["listing_id"] for item in second["recommendations"]]

        assert len(first_ids) == 6
        assert first_ids == second_ids
        assert "CAR-107" in first_ids and "CAR-108" in first_ids
        assert first_ids.index("CAR-107") < first_ids.index("CAR-108")

    def test_fastapi_contract_returns_educational_summary(self):
        assert isinstance(solution.app, FastAPI)
        route = next(
            (
                route
                for route in solution.app.routes
                if getattr(route, "path", None) == "/recommend"
                and "POST" in getattr(route, "methods", set())
            ),
            None,
        )
        assert route is not None
        assert route.endpoint is solution.recommend_endpoint

        solution.configure_workflow(load_history(), load_listings())
        request = solution.RecommendationRequest(
            **profile("NewContractor"),
            query="dealer service history excellent condition",
            top_k=2,
        )
        response = solution.recommend_endpoint(request)
        if inspect.isawaitable(response):
            response = asyncio.run(response)
        body = as_mapping(response)

        assert body["disclaimer"] == DISCLAIMER
        assert body["lower_bound"] >= 0
        assert body["lower_bound"] <= body["predicted_price"] <= body["upper_bound"]
        assert body["eligible_listing_count"] == 6
        assert len(body["recommendations"]) == 2
        assert body["summary"]
        assert all(item["listing_id"].startswith("CAR-") for item in body["recommendations"])
        assert all(
            item["evidence_excerpt"] in item["reason"]
            for item in body["recommendations"]
        )