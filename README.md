# CS611 Assignment 2 — End-to-End ML Pipeline on Airflow

A production-style **credit-default prediction** pipeline orchestrated with Apache Airflow
and packaged with Docker. It builds a medallion datamart (bronze → silver → gold) from raw
loan data, trains an XGBoost model, scores it, and monitors performance & stability over time.

Everything runs inside containers — **no Python, Java, or Spark needed on the host**, just
Docker Desktop.

---

## Pipeline overview

```
data/  (raw CSVs)
   │
   ▼
data_pipeline_bronze_silver_gold ──► train_model ──► model_inference ──► model_monitoring
   (medallion ETL)                   (XGBoost)        (score gold)        (AUC/Gini/PSI)
```

The Airflow DAG (`ml_pipeline`) runs these four tasks in order. Each task executes one of
the project scripts via a `BashOperator`.

| Stage | Script | What it does |
|-------|--------|--------------|
| **Data pipeline** | `test.py` + `utils/` | Bronze ingest → Silver clean → Gold label/feature store (PySpark). Runs only when `run_data_pipeline=true` or the gold store is empty. |
| **Train** | `train.py` | Time-based Train/Dev/OOT split, median impute + target encoding + scaling, XGBoost (with optional `RandomizedSearchCV`), logs curves to TensorBoard, saves a `.pkl` artefact. |
| **Inference** | `inference.py` | Loads the latest model artefact and its preprocessing objects, scores the gold store, writes a gold predictions table. |
| **Monitoring** | `monitoring.py` | Computes per-month AUC/Gini and PSI vs a baseline, writes a gold monitoring table + chart, emits a governance signal. |

---

## Project structure

```
mle-assignment2/
├── dags/
│   └── ml_pipeline_dag.py          # Airflow DAG: data → train → inference → monitor
├── utils/                          # Medallion ETL modules (PySpark)
│   ├── data_processing_bronze_table.py   # raw CSV → bronze (typed ingest)
│   ├── data_processing_silver_table.py   # bronze → silver (clean, dedup, leakage guard)
│   └── data_processing_gold_table.py     # silver → gold label/feature store
├── data/                           # Raw source CSVs (pipeline input)
│   ├── feature_clickstream.csv
│   ├── features_attributes.csv
│   ├── features_financials.csv
│   └── lms_loan_daily.csv
├── test.py                         # Orchestrates the bronze→silver→gold backfill
├── train.py                        # Model training (XGBoost + TensorBoard)
├── inference.py                    # Batch scoring → gold predictions table
├── monitoring.py                   # AUC/Gini/PSI monitoring + chart
├── Dockerfile                      # Airflow image + JDK 17 + Python ML deps
├── docker-compose.yaml             # Postgres + Airflow (init/web/scheduler) + TensorBoard
├── requirements.txt                # Python dependencies
├── data_processing_main.ipynb      # EDA / data-processing notebook
├── data_leakage_check.ipynb        # Leakage-audit notebook
└── test.ipynb                      # Exploratory data-cleaning notebook
```

**Generated at runtime** (git-ignored, rebuilt by the pipeline):
`datamart/` (bronze/silver/gold tables, predictions, monitoring), `model_bank/` (model `.pkl`),
`tensorboard_logs/`, `logs/`.

---

## 1. Build & start
```bash
docker-compose build
docker-compose up
```
Wait until the logs settle (the first build downloads the Airflow image + installs deps).

## 2. Open Airflow
- URL: http://localhost:8080
- Login: **airflow / airflow**

## 3. Run the pipeline
1. Un-pause the **`ml_pipeline`** DAG (toggle on the left).
2. Click **▶ → Trigger DAG w/ config** to set parameters (optional), then **Trigger**.

The DAG runs four tasks in order:
`data_pipeline_bronze_silver_gold → train_model → model_inference → model_monitoring`

### Parameters (in the "Trigger DAG w/ config" form)
| Param | Default | Meaning |
|-------|---------|---------|
| `run_data_pipeline` | false | Rebuild the bronze/silver/gold datamart. (It also auto-builds if the gold store is empty, so the first run always works.) |
| `train_snapshot` | 2024-12-01 | Training snapshot date (Train/Dev/OOT derived backwards). |
| `tune_hyperparameters` | true | true = RandomizedSearchCV; false = train one model with the fixed values below. |
| `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` | 50 / 3 / 0.1 / 0.8 / 0.8 | Used when `tune_hyperparameters = false`. |

## 4. Outputs (written under `datamart/` and `model_bank/` on your host)
- `model_bank/credit_model_*.pkl` — trained model artefact
- `datamart/gold/model_predictions/` — scored predictions (gold table)
- `datamart/gold/model_monitoring/gold_model_monitoring.parquet|csv` — AUC/Gini/PSI by month
- `datamart/gold/model_monitoring/model_monitoring_plot.png` — monitoring chart

## 5. TensorBoard (real-time training curves)
- URL: http://localhost:6006 (after `train_model` has run)

## 6. Stop
```bash
docker-compose down          # keep data
docker-compose down -v       # also wipe the Airflow metadata DB
```

## Notes
- Ports used: **8080** (Airflow), **6006** (TensorBoard), **5432** (Postgres, internal).
- On Linux, file permissions are handled automatically by the `airflow-init` step
  (it chowns the output folders), so no manual `AIRFLOW_UID` change is required.
