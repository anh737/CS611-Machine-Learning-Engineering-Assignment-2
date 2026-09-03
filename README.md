# Credit Default Risk — Orchestrated ML Pipeline

An end-to-end credit-default prediction system: a PySpark medallion datamart, an
XGBoost model, batch scoring, and drift monitoring — wired together as an Airflow
DAG and packaged so the whole thing runs with two Docker commands. No Python,
Java or Spark needed on the host.

The part worth reading is the **leakage fix**. An earlier version of this pipeline
scored well and was wrong: the way gold rows were keyed let information from month
6 of a loan's life into features that a real scoring system would only have on day
one. [Fixing that](#the-leakage-fix) cost accuracy and bought correctness.

> Built for CS611 (Machine Learning Engineering) at Singapore Management University.
> It extends [mle_assignment](https://github.com/anh737/mle_assignment), which
> covers the medallion pipeline alone.

---

## The problem

Predict whether a loan will go bad, where "bad" carries its credit-risk meaning:
**30 or more days past due within the first 6 months on book.** The data is 12,500
customers and 137,500 monthly loan snapshots spanning January 2023 to November 2025.

A model like this is not shipped once. It is retrained, rescored monthly, and
watched for the moment the applicant population drifts away from what it was
trained on — which is why inference and monitoring are pipeline stages here rather
than notebooks.

## Pipeline

```
                        ┌──────────────────────────────────────┐
   data/ (raw CSVs) ──► │  data_pipeline_bronze_silver_gold    │  PySpark
                        │  raw → typed → cleaned → labelled    │
                        └──────────────────┬───────────────────┘
                                           ▼
                        ┌──────────────────────────────────────┐
                        │  train_model                         │  XGBoost
                        │  time-based split, tuned, → .pkl     │  → TensorBoard
                        └──────────────────┬───────────────────┘
                                           ▼
                        ┌──────────────────────────────────────┐
                        │  model_inference                     │  pandas
                        │  score gold → gold predictions table │
                        └──────────────────┬───────────────────┘
                                           ▼
                        ┌──────────────────────────────────────┐
                        │  model_monitoring                    │  AUC / Gini / PSI
                        │  per-month metrics + chart + verdict │
                        └──────────────────────────────────────┘
```

| Stage | Script | What it does |
|---|---|---|
| Data pipeline | `test.py` + `utils/` | Bronze ingest → silver clean → gold label and feature store. Runs when `run_data_pipeline=true`, or automatically when the gold store is empty. |
| Train | `train.py` | Time-based train/dev/OOT split, median imputation, target encoding, scaling, XGBoost with optional `RandomizedSearchCV`. Streams per-round curves to TensorBoard, pickles the artefact. |
| Inference | `inference.py` | Loads the latest artefact with its preprocessing objects, scores the gold store, writes a gold predictions table. |
| Monitoring | `monitoring.py` | Per-month AUC and Gini, PSI against a baseline month, a chart, and a governance verdict. |

## Quickstart

```bash
docker-compose build
docker-compose up
```

Then:

1. Open Airflow at **http://localhost:8080** (login `airflow` / `airflow`).
2. Un-pause the **`ml_pipeline`** DAG.
3. **▶ → Trigger DAG w/ config**, adjust parameters if you want, **Trigger**.

The first run builds the datamart automatically, so a fresh clone works without
any manual setup.

### Parameters

| Param | Default | Meaning |
|---|---|---|
| `run_data_pipeline` | `false` | Rebuild the bronze/silver/gold datamart. Also forced when the gold store is empty. |
| `train_snapshot` | `2024-12-01` | Training snapshot; train/dev/OOT windows are derived backwards from it. |
| `tune_hyperparameters` | `true` | `true` runs `RandomizedSearchCV`; `false` trains once with the fixed values below. |
| `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` | 50 / 3 / 0.1 / 0.8 / 0.8 | Used only when tuning is off. |

### Where the output lands

| Path | Contents |
|---|---|
| `model_bank/credit_model_*.pkl` | model + imputer + target-encoding maps + scaler + training config |
| `datamart/gold/model_predictions/` | scored predictions, one parquet per month |
| `datamart/gold/model_monitoring/gold_model_monitoring.{parquet,csv}` | AUC, Gini and PSI by month |
| `datamart/gold/model_monitoring/model_monitoring_plot.png` | three-panel monitoring chart |

TensorBoard runs at **http://localhost:6006** once `train_model` has executed.

```bash
docker-compose down       # stop, keep data
docker-compose down -v    # also wipe the Airflow metadata DB
```

Ports used: 8080 (Airflow), 6006 (TensorBoard), 5432 (Postgres, internal).

## The leakage fix

The first version of the gold layer keyed each row on the *most recent* snapshot
inside the 1–6 month performance window. Two things went wrong, and only one of
them was visible:

- **Silently visible:** financials and attributes exist only at month 0, the
  application date. Joining them on a month-6 date returned nulls for every
  financial and demographic feature. The model was quietly training on clickstream
  alone.
- **Invisible, and worse:** the clickstream features *did* join — at month 6.
  Those are six months of behaviour that no scoring system would have when the
  application is actually decided. The model was reading the future.

The fix separates the two timelines that were being conflated:

```
   month 0            month 1 ─────────────────► month 6
   ┌──────────┐       ┌─────────────────────────────────┐
   │ FEATURES │       │  LABEL: worst DPD in this window │
   │ anchored │       └─────────────────────────────────┘
   │   here   │
   └──────────┘
```

The label still looks forward — it is the target, so that is legitimate. The gold
row is now anchored on the application date, so every feature joins at the moment
a real system would have it. Two properties fall out of this for free: only loans
observed for the full window get a label (no dependence on when the pipeline
happens to run), and the gold row is identical across partitions, which makes
de-duplication in training trivial.

`data_leakage_check.ipynb` is the audit that surfaced this.

## Other design decisions

**Bronze stores raw, not clean.** Cleaning rules are judgement calls. An untouched
copy means a rule can be revised and the affected months replayed rather than
re-requested from source.

**Target encoding is fitted on train only,** with smoothing, and unseen categories
fall back to the training global mean. The maps are carried inside the model
artefact so inference cannot drift from training.

**Inference uses no Spark.** The artefact carries its own imputer, encoding maps
and scaler, so scoring is pure pandas — faster to run and one less moving part in
the container.

**Class imbalance is handled with `scale_pos_weight`** rather than resampling, so
predicted probabilities stay interpretable as probabilities.

**PSI thresholds follow the standard credit-risk reading:** below 0.10 stable,
0.10–0.25 investigate, above 0.25 refresh the model. `monitoring.py` prints a
verdict rather than only the number, because "PSI = 0.31" is only actionable if
you know where the line sits.

**Training curves go to TensorBoard** via an XGBoost callback on a separate
instrumented copy of the model. The persisted artefact never carries the callback
or a `SummaryWriter`, which would break unpickling.

## Repository layout

```
dags/ml_pipeline_dag.py                   Airflow DAG: data → train → inference → monitor
test.py                                   bronze → silver → gold backfill (the DAG's first task)
train.py                                  training + TensorBoard + model bank
inference.py                              batch scoring → gold predictions
monitoring.py                             AUC / Gini / PSI → gold monitoring table + chart
utils/
  ├── data_processing_bronze_table.py     raw ingest
  ├── data_processing_silver_table.py     cleaning and schema enforcement
  └── data_processing_gold_table.py       labels + feature engineering
data/                                     raw source CSVs (pipeline input)
data_leakage_check.ipynb                  the leakage audit
data_processing_main.ipynb                pipeline walkthrough
test.ipynb                                exploratory data cleaning
Dockerfile, docker-compose.yaml           Airflow + JDK 17 + Python ML deps
```

Generated at runtime and git-ignored: `datamart/`, `model_bank/`,
`tensorboard_logs/`, `logs/`.

## Known limitations

- **`test.py` is the data-pipeline entrypoint, not a test suite.** The name is a
  holdover; `pytest` would try to collect it. Renaming it means updating one
  `BashOperator` in the DAG.
- **No automated tests.** Correctness rests on the notebook audits, which is
  exactly the kind of thing the leakage bug slipped through.
- **Roughly 28% of customers have no clickstream rows**, so `fe_1`–`fe_20` are
  missing rather than zero for them.
- **PSI is baselined on the earliest scored month**, not the training-period score
  distribution. It is reproducible, but it is a proxy.
- **The raw CSVs are committed** (~36 MB). Convenient for a grader cloning the
  repo; not what you would do with a real feed.
- **The model bank keeps every artefact and inference picks the newest by mtime.**
  There is no promotion gate or champion/challenger comparison.

## Stack

Apache Airflow 2.9 · PySpark 3.5 · XGBoost · scikit-learn · pandas · TensorBoard ·
Postgres · Docker Compose

## License

MIT — see [LICENSE](LICENSE).
