"""Bronze layer: land each raw source file as a dated partition.

No cleaning happens here by design — bronze is an as-received copy of the source
systems, filtered to one monthly snapshot so the layer stays replayable.
"""

import os
from datetime import datetime

from pyspark.sql.functions import col


def process_bronze_table(snapshot_date_str, bronze_base_directory, spark):
    """Ingest every raw source file for one snapshot date into the bronze tier.

    Args:
        snapshot_date_str: Snapshot to ingest, ``YYYY-MM-DD``.
        bronze_base_directory: Root bronze path, e.g. ``datamart/bronze/``.
        spark: Active Spark session.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    suffix = snapshot_date_str.replace("-", "_")

    print(f"[bronze] {snapshot_date_str}")

    # 1. LMS loan daily — the operational loan book, source of the labels.
    lms_raw_path = "data/lms_loan_daily.csv"
    df_lms = spark.read.csv(lms_raw_path, header=True, inferSchema=True) \
                  .filter(col('snapshot_date') == snapshot_date)
    print(f"  lms: {df_lms.count()} rows")

    lms_dir = os.path.join(bronze_base_directory, "lms")
    os.makedirs(lms_dir, exist_ok=True)
    lms_filepath = os.path.join(lms_dir, f"bronze_loan_daily_{suffix}.csv")
    df_lms.toPandas().to_csv(lms_filepath, index=False)

    # 2. Feature attributes — customer demographics.
    attr_raw_path = "data/features_attributes.csv"
    df_attr = spark.read.csv(attr_raw_path, header=True, inferSchema=True) \
                  .filter(col('snapshot_date') == snapshot_date)
    print(f"  attributes: {df_attr.count()} rows")

    attr_dir = os.path.join(bronze_base_directory, "attributes")
    os.makedirs(attr_dir, exist_ok=True)
    attr_filepath = os.path.join(attr_dir, f"bronze_attributes_{suffix}.csv")
    df_attr.toPandas().to_csv(attr_filepath, index=False)

    # 3. Feature financials — income, debt and repayment behaviour.
    fin_raw_path = "data/features_financials.csv"
    df_fin = spark.read.csv(fin_raw_path, header=True, inferSchema=True) \
                  .filter(col('snapshot_date') == snapshot_date)
    print(f"  financials: {df_fin.count()} rows")

    fin_dir = os.path.join(bronze_base_directory, "financials")
    os.makedirs(fin_dir, exist_ok=True)
    fin_filepath = os.path.join(fin_dir, f"bronze_financials_{suffix}.csv")
    df_fin.toPandas().to_csv(fin_filepath, index=False)

    # 4. Feature clickstream — 20 anonymised behavioural event counters.
    click_raw_path = "data/feature_clickstream.csv"
    df_click = spark.read.csv(click_raw_path, header=True, inferSchema=True) \
                  .filter(col('snapshot_date') == snapshot_date)
    print(f"  clickstream: {df_click.count()} rows")

    click_dir = os.path.join(bronze_base_directory, "clickstream")
    os.makedirs(click_dir, exist_ok=True)
    click_filepath = os.path.join(click_dir, f"bronze_clickstream_{suffix}.csv")
    df_click.toPandas().to_csv(click_filepath, index=False)

    return df_lms
