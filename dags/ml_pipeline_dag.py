"""
CS611 Assignment 2 - End-to-end ML pipeline DAG.

Pipeline graph:
    data_pipeline (bronze -> silver -> gold)   [runs only if run_data_pipeline=True]
        -> train_model        [hyperparameters configurable via params]
            -> model_inference
                -> model_monitoring

All stages run the existing project scripts via BashOperator from PROJECT_DIR
so their relative paths (data/, datamart/, model_bank/, tensorboard_logs/) work.

The DAG is parameterised: open "Trigger DAG w/ config" in the Airflow UI to set
the parameters below BEFORE a run.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.models.param import Param

# Directory inside the container where the whole project is mounted
PROJECT_DIR = "/opt/airflow/project"

default_args = {
    "owner": "cs611",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# -----------------------------------------------------------------------------
# DAG-level parameters (shown in the "Trigger DAG w/ config" form)
# -----------------------------------------------------------------------------
dag_params = {
    # Gate for the data pipeline: only (re)build the datamart when explicitly asked.
    "run_data_pipeline": Param(
        False, type="boolean",
        title="Run data pipeline (bronze/silver/gold)",
        description="If TRUE, rebuild the datamart before training. If FALSE, reuse the existing datamart.",
    ),
    # Training snapshot date (Train/Dev/OOT windows are derived backwards from it).
    "train_snapshot": Param(
        "2024-12-01", type="string",
        title="Training snapshot date (YYYY-MM-DD)",
    ),
    # Hyperparameter controls for the training task.
    "tune_hyperparameters": Param(
        True, type="boolean",
        title="Tune hyperparameters",
        description="If TRUE, run RandomizedSearchCV. If FALSE, train one model with the fixed values below.",
    ),
    "n_estimators": Param(50, type="integer", title="n_estimators (when tuning is OFF)"),
    "max_depth": Param(3, type="integer", title="max_depth (when tuning is OFF)"),
    "learning_rate": Param(0.1, type="number", title="learning_rate (when tuning is OFF)"),
    "subsample": Param(0.8, type="number", title="subsample (when tuning is OFF)"),
    "colsample_bytree": Param(0.8, type="number", title="colsample_bytree (when tuning is OFF)"),
}

with DAG(
    dag_id="ml_pipeline",
    description="End-to-end credit-default ML pipeline (data -> train -> inference -> monitor)",
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,   # triggered manually (with config)
    catchup=False,
    params=dag_params,
    tags=["cs611", "assignment2", "ml-pipeline"],
) as dag:

    # -- TIER 1-3: data pipeline, gated by the run_data_pipeline param ---------
    # The task always succeeds; it only executes test.py when the flag is True,
    # otherwise it prints a skip message and the existing datamart is reused.
    data_pipeline = BashOperator(
        task_id="data_pipeline_bronze_silver_gold",
        # Runs test.py when run_data_pipeline=True, OR automatically when the gold
        # label store is empty (so a brand-new checkout always builds the datamart
        # on the first run). Otherwise it skips and reuses the existing datamart.
        bash_command=(
            f'cd {PROJECT_DIR} && '
            'if [ "{{ params.run_data_pipeline }}" = "True" ] || '
            '[ -z "$(find datamart/gold/label_store -name \'*.parquet\' 2>/dev/null | head -1)" ]; then '
            'echo "[DAG] Building bronze/silver/gold (test.py)"; python test.py; '
            'else '
            'echo "[DAG] run_data_pipeline=False and gold exists -> reusing existing datamart"; '
            'fi'
        ),
    )

    # -- Model training: hyperparameters come from the DAG params --------------
    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f'cd {PROJECT_DIR} && python train.py '
            '--snapshot {{ params.train_snapshot }} '
            '--tune {{ params.tune_hyperparameters }} '
            '--n_estimators {{ params.n_estimators }} '
            '--max_depth {{ params.max_depth }} '
            '--learning_rate {{ params.learning_rate }} '
            '--subsample {{ params.subsample }} '
            '--colsample_bytree {{ params.colsample_bytree }}'
        ),
    )

    # -- Inference: score the gold feature store, write gold predictions table -
    model_inference = BashOperator(
        task_id="model_inference",
        bash_command=f"cd {PROJECT_DIR} && python inference.py",
    )

    # -- Monitoring: AUC/Gini + PSI over time, write gold monitoring table + PNG
    model_monitoring = BashOperator(
        task_id="model_monitoring",
        bash_command=f"cd {PROJECT_DIR} && python monitoring.py",
    )

    data_pipeline >> train_model >> model_inference >> model_monitoring
