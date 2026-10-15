# Eligible-Price Used-Car Recommendation Workflow

## Purpose

Build an educational shopping workflow that predicts a car loan-eligible price range from five pre-decision profile fields, filters used-car listings to that range, and ranks the eligible listings using their unstructured service and condition evidence.

This project addresses REQ-001, REQ-002, and REQ-003. It is not a credit-approval system and must not describe its result as an approval or financial advice.

## Repository layout

- `practice_project/`: learner workspace. Implement `app.py` here.
- `reference_project/`: separately maintained completed implementation.
- `validation/`: immutable pytest validation and deterministic datasets.
- `guided-instructions/`: this guide.

Run validation from the repository root:

```text
python -m pytest -q
```

## Simulation 1 — guided implementation (1.75 hours)

### Step 1: Load and validate the borrower history

Read `validation/fixtures/borrower_history.csv` with pandas. The model target is `eligible_amount`.

The model's complete and ordered feature list must be:

1. `income`
2. `credit_score`
3. `employment_type`
4. `loan_tenure_months`
5. `down_payment`

The fixture deliberately contains `purchased_vehicle_price`, `approved_interest_rate`, and `final_financed_amount`. These are post-selection or post-approval leakage fields. Do not place them in the preprocessing or regression pipeline.

Use the first 18 rows for training and the final 6 rows for deterministic holdout validation. Configure categorical preprocessing so missing and previously unseen employment values can be scored without refitting or raising an error.

### Step 2: Build the regression workflow

Implement `train_workflow(history, listings)`. It must return a fitted workflow object exposing:

- `feature_columns`
- `target_column`
- `text_fields`
- `validation_report`
- `predict_range(profile)`
- `recommend(profile, query, top_k)`

Use a scikit-learn preprocessing-and-regression pipeline. `predict_range` must return numeric `predicted_price`, `lower_bound`, and `upper_bound`. Construct the bounds as:

- `lower_bound = max(0, predicted_price * 0.85)`
- `upper_bound = predicted_price * 1.15`

The ordering `0 <= lower_bound <= predicted_price <= upper_bound` must always hold.

### Step 3: Produce validation evidence

Compute holdout mean absolute error for the fitted regression model and for a baseline that always predicts the training-target median. The validation report must include:

- `holdout_count`
- `model_mae`
- `baseline_mae`
- `range_coverage`
- `segments`

`range_coverage` is the fraction of holdout targets inside their predicted 15% bounds. `segments` must contain diagnostics for `income_band`, `credit_score_band`, and `employment_type`. Each segment entry must report `count`, `mae`, and `coverage`; counts for each dimension must total the six holdout records.

### Step 4: Implement bounded semantic retrieval

Read `validation/fixtures/car_listings.csv`. Apply the predicted lower and upper bounds as a hard price filter before fitting or applying text similarity for the request.

Build retrieval text from all three fields:

- `service_history`
- `condition_notes`
- `other_details`

Use scikit-learn text vectorization and cosine similarity. Sort by descending similarity and then ascending `listing_id` to resolve equal-score ties deterministically.

Each returned recommendation must contain:

- `rank`
- `listing_id`
- `price`
- `similarity_score`
- `evidence_excerpt`
- `reason`

The evidence excerpt must be copied exactly from one of that listing's three source text fields. The reason must cite that excerpt rather than inventing vehicle facts. Return `eligible_listing_count` before applying `top_k`.

### Step 5: Expose the FastAPI contract

In `practice_project/app.py`, provide:

- a FastAPI instance named `app`
- `configure_workflow(history, listings)` to install a fitted workflow for serving
- a Pydantic request type named `RecommendationRequest`
- a callable named `recommend_endpoint`
- a `POST /recommend` route backed by that callable

The request fields are `income`, `credit_score`, `employment_type`, `loan_tenure_months`, `down_payment`, `query`, and `top_k`.

The response must include profile-derived bounds, eligible listing count, ranked recommendations, a grounded summary, and this exact disclaimer:

`Educational shopping estimate only; not a credit approval or financial advice.`

## Practical Assessment 1 — independent extension (1.25 hours)

Without changing validation files, complete the practice implementation and run the full suite. Submit the working `practice_project/app.py`.

Pass requires all three components:

1. The fitted model uses exactly the five pre-decision features, excludes all supplied leakage fields, accepts missing and unfamiliar employment categories, and returns ordered non-negative bounds.
2. The fitted model beats the supplied median baseline and reports holdout range coverage plus income, credit-score, and employment segment diagnostics.
3. Every recommendation is within the predicted bounds and grounded in a source listing excerpt, with price filtering occurring before text ranking. This is the critical gate.

All three components, including the critical retrieval-and-grounding gate, must pass.