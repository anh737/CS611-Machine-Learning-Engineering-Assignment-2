"""Run the medallion backfill end to end: bronze -> silver -> gold.

This is the script the Airflow task ``data_pipeline_bronze_silver_gold`` calls.

Usage:
    python test.py                            # backfill every month in the default window
    python test.py --snapshotdate 2024-06-01  # process a single month
"""

import os
import glob
import argparse
from datetime import datetime

import pyspark

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


def generate_first_of_month_dates(start_date_str, end_date_str):
    """Return the first day of each calendar month in [start, end] as YYYY-MM-DD."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates


def main(snapshot_date_arg=None):
    spark = pyspark.sql.SparkSession.builder \
        .appName("medallion-backfill") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    bronze_base_directory = "datamart/bronze/"
    silver_base_directory = "datamart/silver/"
    gold_label_store_directory = "datamart/gold/label_store/"

    for directory in (bronze_base_directory, silver_base_directory, gold_label_store_directory):
        os.makedirs(directory, exist_ok=True)

    if snapshot_date_arg:
        dates_str_lst = [snapshot_date_arg]
        print(f"Single-month run: {snapshot_date_arg}")
    else:
        start_date_str = "2023-01-01"
        end_date_str = "2025-11-01"
        dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
        print(f"Backfill window: {dates_str_lst[0]} to {dates_str_lst[-1]} ({len(dates_str_lst)} months)")

    # --- Bronze: land the raw source files, one partition per month ---
    print("\n== Bronze: raw ingest ==")
    for date_str in dates_str_lst:
        utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_base_directory, spark)

    # --- Silver: enforce schema, clean junk values, dedupe ---
    print("\n== Silver: clean and type ==")
    for date_str in dates_str_lst:
        utils.data_processing_silver_table.process_silver_table(
            date_str, bronze_base_directory, silver_base_directory, spark
        )

    # --- Gold: build the label store (30 DPD within 6 MOB) ---
    print("\n== Gold: label store ==")
    for date_str in dates_str_lst:
        utils.data_processing_gold_table.process_labels_gold_table(
            date_str, gold_label_store_directory, spark, mob=6, dpd=30
        )

    files_list = glob.glob(os.path.join(gold_label_store_directory, "*"))
    if files_list:
        df = spark.read.option("header", "true").parquet(*files_list)
        print(f"\nGold label store row count: {df.count()}")
        df.show()
    else:
        print("\nNo gold partitions written — check that silver holds loans matured past the MOB window.")

    spark.stop()
    print("\nBackfill complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medallion backfill runner (bronze -> silver -> gold)"
    )
    parser.add_argument(
        "--snapshotdate",
        type=str,
        required=False,
        help="Process a single month (YYYY-MM-DD). Omit to backfill the full window.",
    )
    args = parser.parse_args()
    main(snapshot_date_arg=args.snapshotdate)
