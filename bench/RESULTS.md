# Benchmark results

provider=anthropic model=claude-sonnet-5-5

| dataset | model family | LLM calls (failed) | L2 proposed → applied / rejected | L4 proposed → validated / rejected | L5 leak flags | replay exact |
|---|---|---|---|---|---|---|
| titanic | tree | 2 (0) | 10 → 7 / 3 | 12 → 12 / 0 | boat, body | yes |
| titanic | linear | 2 (0) | 7 → 7 / 0 | 14 → 14 / 0 | boat, body | yes |
| bank_marketing | tree | 2 (0) | 7 → 7 / 0 | 29 → 29 / 0 | - | yes |
| bank_marketing | linear | 2 (0) | 7 → 7 / 0 | 27 → 27 / 0 | - | yes |
| adult_planted | tree | 2 (0) | 3 → 3 / 0 | 18 → 18 / 0 | final_review | yes |
| adult_planted | linear | 2 (0) | 6 → 6 / 0 | 20 → 20 / 0 | final_review | yes |
| telco_churn | tree | 2 (0) | 1 → 1 / 0 | 22 → 21 / 1 | - | yes |
| telco_churn | linear | 2 (0) | 1 → 1 / 0 | 28 → 28 / 0 | - | yes |
| boston_housing | tree | 2 (0) | 8 → 8 / 0 | 17 → 17 / 0 | - | yes |
| boston_housing | linear | 2 (0) | 7 → 7 / 0 | 31 → 25 / 6 | - | yes |
| hotel_bookings | tree | 2 (0) | 15 → 15 / 0 | 27 → 27 / 0 | reservation_status | yes |
| hotel_bookings | linear | 2 (0) | 12 → 12 / 0 | 35 → 35 / 0 | reservation_status | yes |
| hr_attrition | tree | 2 (0) | 10 → 10 / 0 | 20 → 20 / 0 | - | yes |
| hr_attrition | linear | 2 (0) | 10 → 10 / 0 | 42 → 42 / 0 | - | yes |
| pima_diabetes | tree | 2 (0) | 8 → 8 / 0 | 13 → 13 / 0 | - | yes |
| pima_diabetes | linear | 2 (0) | 8 → 8 / 0 | 21 → 18 / 3 | - | yes |
| stroke | tree | 2 (0) | 4 → 4 / 0 | 21 → 21 / 0 | - | yes |
| stroke | linear | 2 (0) | 4 → 4 / 0 | 18 → 16 / 2 | - | yes |
| california_housing | tree | 2 (0) | 6 → 6 / 0 | 18 → 17 / 1 | - | yes |
| california_housing | linear | 2 (0) | 6 → 6 / 0 | 19 → 19 / 0 | - | yes |

**Total LLM proposals: 592. Rejected by deterministic code: 16** (L2 3/140, L4 13/452).

Rejection reasons (L2): {"Imputing 77% missing values would overwrite most of 'cabin' with one value; use a missing indicator or drop it": 1, "Imputing 63% missing values would overwrite most of 'boat' with one value; use a missing indicator or drop it": 1, "Imputing 91% missing values would overwrite most of 'body' with one value; use a missing indicator or drop it": 1}

Rejection reasons (L4): {"Source columns not found": 13}

### Held-out model score

AgentPrep runs on an 80% training split. Its saved pipeline (`pipeline_<run_id>.json`) is then applied to the untouched 20% test split, and a model trained on the training rows is scored on the test rows. Both arms get the same minimal sklearn prep fitted on train (median imputation; ordinal codes for trees, scaling and one-hot for linear models), so the gap between them is what AgentPrep adds.

"Known leaks removed" drops the documented leak columns and any feature built from them, which isolates feature value from leak handling. Raw scores include leaks and are inflated where leaks exist.

| dataset | model | metric | raw | raw, known leaks removed | AgentPrep | AgentPrep, known leaks removed | feature lift |
|---|---|---|---|---|---|---|---|
| titanic | gradient boosting | ROC AUC | 0.994 | 0.857 | 0.836 | 0.836 | -0.020 |
| titanic | linear | ROC AUC | 0.995 | 0.876 | 0.876 | 0.876 | +0.000 |
| bank_marketing | gradient boosting | ROC AUC | 0.934 | 0.798 | 0.937 | 0.802 | +0.004 |
| bank_marketing | linear | ROC AUC | 0.907 | 0.769 | 0.911 | 0.776 | +0.007 |
| adult_planted | gradient boosting | ROC AUC | 1.000 | 0.925 | 0.924 | 0.924 | -0.001 |
| adult_planted | linear | ROC AUC | 1.000 | 0.901 | 0.906 | 0.906 | +0.004 |
| telco_churn | gradient boosting | ROC AUC | 0.825 | 0.825 | 0.830 | 0.830 | +0.005 |
| telco_churn | linear | ROC AUC | 0.852 | 0.852 | 0.858 | 0.858 | +0.007 |
| boston_housing | gradient boosting | R2 | 0.707 | 0.707 | 0.720 | 0.720 | +0.013 |
| boston_housing | linear | R2 | 0.588 | 0.588 | 0.708 | 0.708 | +0.120 |
| hotel_bookings | gradient boosting | ROC AUC | 1.000 | 0.950 | 0.988 | 0.949 | -0.000 |
| hotel_bookings | linear | ROC AUC | 1.000 | 0.898 | 0.897 | 0.895 | -0.003 |
| hr_attrition | gradient boosting | ROC AUC | 0.781 | 0.781 | 0.750 | 0.750 | -0.030 |
| hr_attrition | linear | ROC AUC | 0.809 | 0.809 | 0.815 | 0.815 | +0.006 |
| pima_diabetes | gradient boosting | ROC AUC | 0.840 | 0.840 | 0.826 | 0.826 | -0.014 |
| pima_diabetes | linear | ROC AUC | 0.884 | 0.884 | 0.873 | 0.873 | -0.011 |
| stroke | gradient boosting | ROC AUC | 0.791 | 0.791 | 0.811 | 0.811 | +0.020 |
| stroke | linear | ROC AUC | 0.804 | 0.804 | 0.819 | 0.819 | +0.015 |
| california_housing | gradient boosting | R2 | 0.830 | 0.830 | 0.852 | 0.852 | +0.023 |
| california_housing | linear | R2 | 0.638 | 0.638 | 0.666 | 0.666 | +0.028 |

### LLM-suspected leaks (surfaced as warnings, never auto-dropped)

- titanic (tree): `boat` (Lifeboat assignment is recorded only for those who boarded a lifeboat during the sinking, which is strongly tied to survival and not known before the outcome.), `body` (Body identification number is assigned only to people who died and whose bodies were recovered, so it directly encodes the outcome.)
- titanic (linear): `boat` (The lifeboat assignment is recorded during or after the sinking and is directly tied to survival. Passengers without a boat mostly died, and missingness is itself informative.), `body` (A body identification number exists only for passengers who died and whose bodies were recovered, so it encodes the outcome directly.)
- bank_marketing (tree): `duration` (Call duration is only known after the call ends, and it is strongly tied to the outcome (a longer call usually means the client subscribed). It would not be available when predicting before the call.)
- bank_marketing (linear): `duration` (Call duration is only known after the call ends, and it is strongly tied to the outcome (a longer call usually means the client subscribed). It would not be available when deciding whom to call.)
- adult_planted (tree): `final_review` (This is not a standard census attribute. A binary 'final review' field is likely assigned after the income outcome is known or decided, so it would not be available at prediction time. Verify its orig)
- adult_planted (linear): `final_review` (This column is not part of the standard census features. It looks like a post-hoc review or label-derived field that would be recorded after the income outcome is known, so it would not be available a)
- telco_churn (tree): none
- telco_churn (linear): none
- boston_housing (tree): none
- boston_housing (linear): none
- hotel_bookings (tree): `reservation_status` (Holds the final booking outcome (Canceled, Check-Out or No-Show), which directly encodes the target and is only known after the outcome.), `reservation_status_date` (Date the final reservation status was set. It is recorded at or after cancellation or checkout, so it is not known when a prediction would be made.), `assigned_room_type` (Assigned at or after check-in, and often differs from the reserved room only for stays that actually happen, so it is tied to the outcome.), `adr` (Average daily rate can be finalized or adjusted after the stay, and canceled bookings may carry different values, so it may be influenced by the outcome.), `required_car_parking_spaces` (Parking is typically recorded at or after arrival, so it is mostly known only for guests who actually stay.), `booking_changes` (Changes can accumulate up to check-in or cancellation, so the final count may not be known at prediction time.)
- hotel_bookings (linear): `reservation_status` (Values Canceled, No-Show and Check-Out directly encode the cancellation outcome and are only known after the booking result.), `reservation_status_date` (This is the date the final status was set. For cancellations it is the cancellation date, so it is recorded because of the outcome.), `assigned_room_type` (The room is often assigned at or after check-in, and it is typically missing or changed for canceled bookings, so it may reflect the outcome.), `booking_changes` (The count accumulates up to check-in or cancellation, so the final value may not be known at prediction time.), `days_in_waiting_list` (It is finalized only once the booking is confirmed or canceled.)
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
| titanic | linear | boat | caught by deterministic leak detector |
| titanic | linear | body | caught by deterministic leak detector |
| bank_marketing | tree | duration | surfaced as LLM-suspected leak (warning, kept) |
| bank_marketing | linear | duration | surfaced as LLM-suspected leak (warning, kept) |
| adult_planted | tree | final_review | caught by deterministic leak detector |
| adult_planted | linear | final_review | caught by deterministic leak detector |
| hotel_bookings | tree | reservation_status | caught by deterministic leak detector |
| hotel_bookings | tree | reservation_status_date | surfaced as LLM-suspected leak (warning, kept) |
| hotel_bookings | linear | reservation_status | caught by deterministic leak detector |
| hotel_bookings | linear | reservation_status_date | surfaced as LLM-suspected leak (warning, kept) |
