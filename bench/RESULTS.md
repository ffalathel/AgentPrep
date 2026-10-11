# Benchmark results

provider=anthropic model=claude-sonnet-5-5

Each of the 20 runs was executed in its own process.

| dataset | model family | LLM calls (failed) | L2 proposed → applied / rejected | L4 proposed → validated / rejected | L5 leak flags | replay exact |
|---|---|---|---|---|---|---|
| titanic | tree | 2 (0) | 12 → 9 / 3 | 13 → 13 / 0 | boat, body | yes |
| titanic | linear | 2 (0) | 13 → 13 / 0 | 13 → 12 / 1 | - | yes |
| bank_marketing | tree | 2 (0) | 9 → 9 / 0 | 27 → 27 / 0 | - | yes |
| bank_marketing | linear | 2 (0) | 9 → 9 / 0 | 32 → 24 / 8 | - | yes |
| adult_planted | tree | 2 (0) | 11 → 11 / 0 | 19 → 19 / 0 | final_review | yes |
| adult_planted | linear | 2 (0) | 12 → 12 / 0 | 19 → 19 / 0 | final_review | yes |
| telco_churn | tree | 2 (0) | 1 → 1 / 0 | 22 → 22 / 0 | - | yes |
| telco_churn | linear | 2 (0) | 1 → 1 / 0 | 28 → 26 / 2 | - | yes |
| boston_housing | tree | 2 (0) | 7 → 7 / 0 | 18 → 18 / 0 | - | yes |
| boston_housing | linear | 2 (0) | 7 → 7 / 0 | 23 → 21 / 2 | - | yes |
| hotel_bookings | tree | 2 (0) | 14 → 14 / 0 | 22 → 21 / 1 | reservation_status, reservation_status_date | yes |
| hotel_bookings | linear | 2 (0) | 14 → 14 / 0 | 34 → 34 / 0 | reservation_status, reservation_status_date | yes |
| hr_attrition | tree | 2 (0) | 13 → 13 / 0 | 23 → 23 / 0 | - | yes |
| hr_attrition | linear | 2 (0) | 14 → 14 / 0 | 39 → 39 / 0 | - | yes |
| pima_diabetes | tree | 2 (0) | 17 → 16 / 1 | 17 → 17 / 0 | - | yes |
| pima_diabetes | linear | 2 (0) | 18 → 17 / 1 | 30 → 23 / 7 | - | yes |
| stroke | tree | 2 (0) | 6 → 6 / 0 | 22 → 22 / 0 | - | yes |
| stroke | linear | 2 (0) | 6 → 6 / 0 | 20 → 20 / 0 | - | yes |
| california_housing | tree | 2 (0) | 6 → 6 / 0 | 19 → 18 / 1 | - | yes |
| california_housing | linear | 2 (0) | 6 → 6 / 0 | 20 → 20 / 0 | - | yes |

**Total LLM proposals: 656. Rejected by deterministic code: 27** (L2 5/196, L4 22/460).

Rejection reasons (L2): {"Imputing 77% missing values would overwrite most of 'cabin' with one value; use a missing indicator or drop it": 1, "Imputing 63% missing values would overwrite most of 'boat' with one value; use a missing indicator or drop it": 1, "Imputing 91% missing values would overwrite most of 'body' with one value; use a missing indicator or drop it": 1, "Imputing 51% missing values would overwrite most of 'insulin' with one value; use a missing indicator or drop it": 2}

Rejection reasons (L4): {"Source columns not found": 22}

### Held-out model score

AgentPrep runs on an 80% training split. Its saved pipeline (`pipeline_<run_id>.json`) is then applied to the untouched 20% test split, and a model trained on the training rows is scored on the test rows. Both arms get the same minimal sklearn prep fitted on train (median imputation; ordinal codes for trees, scaling and one-hot for linear models), so the gap between them is what AgentPrep adds.

"Known leaks removed" drops the documented leak columns and any feature built from them, which isolates feature value from leak handling. Raw scores include leaks and are inflated where leaks exist.

| dataset | model | metric | raw | raw, known leaks removed | AgentPrep | AgentPrep, known leaks removed | feature lift |
|---|---|---|---|---|---|---|---|
| titanic | gradient boosting | ROC AUC | 0.994 | 0.857 | 0.834 | 0.834 | -0.023 |
| titanic | linear | ROC AUC | 0.995 | 0.876 | 0.865 | 0.865 | -0.011 |
| bank_marketing | gradient boosting | ROC AUC | 0.934 | 0.798 | 0.936 | 0.800 | +0.002 |
| bank_marketing | linear | ROC AUC | 0.907 | 0.769 | 0.911 | 0.775 | +0.006 |
| adult_planted | gradient boosting | ROC AUC | 1.000 | 0.925 | 0.924 | 0.924 | -0.001 |
| adult_planted | linear | ROC AUC | 1.000 | 0.901 | 0.900 | 0.900 | -0.001 |
| telco_churn | gradient boosting | ROC AUC | 0.825 | 0.825 | 0.822 | 0.822 | -0.004 |
| telco_churn | linear | ROC AUC | 0.852 | 0.852 | 0.858 | 0.858 | +0.006 |
| boston_housing | gradient boosting | R2 | 0.707 | 0.707 | 0.692 | 0.692 | -0.015 |
| boston_housing | linear | R2 | 0.588 | 0.588 | 0.686 | 0.686 | +0.098 |
| hotel_bookings | gradient boosting | ROC AUC | 1.000 | 0.950 | 0.949 | 0.949 | -0.000 |
| hotel_bookings | linear | ROC AUC | 1.000 | 0.898 | 0.895 | 0.895 | -0.003 |
| hr_attrition | gradient boosting | ROC AUC | 0.781 | 0.781 | 0.785 | 0.785 | +0.005 |
| hr_attrition | linear | ROC AUC | 0.809 | 0.809 | 0.802 | 0.802 | -0.007 |
| pima_diabetes | gradient boosting | ROC AUC | 0.840 | 0.840 | 0.826 | 0.826 | -0.014 |
| pima_diabetes | linear | ROC AUC | 0.884 | 0.884 | 0.882 | 0.882 | -0.002 |
| stroke | gradient boosting | ROC AUC | 0.791 | 0.791 | 0.813 | 0.813 | +0.022 |
| stroke | linear | ROC AUC | 0.804 | 0.804 | 0.822 | 0.822 | +0.017 |
| california_housing | gradient boosting | R2 | 0.830 | 0.830 | 0.843 | 0.843 | +0.013 |
| california_housing | linear | R2 | 0.638 | 0.638 | 0.667 | 0.667 | +0.029 |

### LLM-suspected leaks (surfaced as warnings, never auto-dropped)

- titanic (tree): `boat` (Lifeboat assignment is recorded during or after the sinking; only survivors boarded boats, so it directly encodes the outcome.), `body` (Body identification number is assigned only to those who died and whose bodies were recovered, so it is a direct consequence of the target.)
- titanic (linear): `boat` (Lifeboat assignment occurs during the sinking and is essentially only recorded for survivors, so it reveals the outcome.), `body` (Body identification number is assigned only to those who died and whose bodies were recovered, so it is determined by the outcome.)
- bank_marketing (tree): `duration` (Call duration is only known after the call ends, and it is strongly tied to the outcome. It would not be available when predicting whether a client subscribes.)
- bank_marketing (linear): `duration` (Call duration is only known after the call ends, and it is strongly tied to the outcome (a long call usually means a subscription). It would not be available when predicting before the call.)
- adult_planted (tree): `final_review` (This column is not part of the standard census features. It looks like a review or label recorded after the income outcome was determined, so it would not be available at prediction time.)
- adult_planted (linear): `final_review` (It is not a standard demographic or employment attribute and looks like a review or decision recorded after, or because of, the income outcome. It would likely not be available at prediction time.)
- telco_churn (tree): none
- telco_churn (linear): none
- boston_housing (tree): none
- boston_housing (linear): none
- hotel_bookings (tree): `reservation_status` (It records the final outcome (Check-Out, Canceled, No-Show), which directly encodes the cancellation target and is only known after the outcome.), `reservation_status_date` (It is the date the reservation status was last set. For cancelled bookings this is the cancellation date, so it is only known after the outcome.), `assigned_room_type` (It is assigned at or after check-in and can differ from the reserved type. Cancelled bookings are never assigned a room, so it may reflect the outcome.), `booking_changes` (Changes can accumulate up to check-in or cancellation, so the count may be recorded after the prediction time.), `required_car_parking_spaces` (Parking is typically only recorded for guests who actually arrive, so it is tied to the outcome.)
- hotel_bookings (linear): `reservation_status` (Records the final outcome (Canceled, Check-Out, No-Show), which directly encodes the target and is only known after the outcome.), `reservation_status_date` (The date the status was last set, i.e. the cancellation or checkout date. It is only known after the outcome.), `assigned_room_type` (Assigned at check-in or after the booking is finalized. It is typically unavailable for canceled bookings and may be influenced by the outcome.), `booking_changes` (Accumulates over the booking's life up to check-in or cancellation, so it may not be fully known at prediction time.)
- hr_attrition (tree): none
- hr_attrition (linear): none
- pima_diabetes (tree): none
- pima_diabetes (linear): none
- stroke (tree): none
- stroke (linear): none
- california_housing (tree): none
- california_housing (linear): none

### Documented leaks

| dataset | model family | column | outcome |
|---|---|---|---|
| titanic | tree | boat | caught by deterministic leak detector |
| titanic | tree | body | caught by deterministic leak detector |
| titanic | linear | boat | removed by LLM drop action |
| titanic | linear | body | removed by LLM drop action |
| bank_marketing | tree | duration | surfaced as LLM-suspected leak (warning, kept) |
| bank_marketing | linear | duration | surfaced as LLM-suspected leak (warning, kept) |
| adult_planted | tree | final_review | caught by deterministic leak detector |
| adult_planted | linear | final_review | caught by deterministic leak detector |
| hotel_bookings | tree | reservation_status | caught by deterministic leak detector |
| hotel_bookings | tree | reservation_status_date | caught by deterministic leak detector |
| hotel_bookings | linear | reservation_status | caught by deterministic leak detector |
| hotel_bookings | linear | reservation_status_date | caught by deterministic leak detector |
