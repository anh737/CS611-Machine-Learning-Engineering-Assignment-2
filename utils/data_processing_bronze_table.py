import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


# def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark):
#     # prepare arguments
#     snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
#     # connect to source back end - IRL connect to back end source system
#     csv_file_path = "data/lms_loan_daily.csv"

#     # load data - IRL ingest from back end source system
#     df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
#     print(snapshot_date_str + 'row count:', df.count())
    
#     # save bronze table to datamart - IRL connect to database to write
#     partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
#     filepath = bronze_lms_directory + partition_name
#     df.toPandas().to_csv(filepath, index=False)
#     print('saved to:', filepath)

#     return df


def process_bronze_table(snapshot_date_str, bronze_base_directory, spark):
    """
    Ingest all raw source files into partitioned folders under the Bronze tier.
    Expected base directory path structure: "datamart/bronze/"
    """
    # Parse date parameters and format filesystem suffixes
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    suffix = snapshot_date_str.replace('-', '_')
    
    print(f"\n--- [BRONZE TIER] Commencing ingestion loop for date: {snapshot_date_str} ---")

    # =========================================================================
    # 1. PROCESSING: LMS LOAN DAILY (Isolate target operational snapshots)
    # =========================================================================
    lms_raw_path = "data/lms_loan_daily.csv"
    df_lms = spark.read.csv(lms_raw_path, header=True, inferSchema=True) \
                  .filter(col('snapshot_date') == snapshot_date)
    print(f"  - LMS Loan Daily | Extracted payload count: {df_lms.count()}")
    
    lms_dir = os.path.join(bronze_base_directory, "lms")
    os.makedirs(lms_dir, exist_ok=True)
    lms_filepath = os.path.join(lms_dir, f"bronze_loan_daily_{suffix}.csv")
    df_lms.toPandas().to_csv(lms_filepath, index=False)
    print(f"  -> Stored trace file: {lms_filepath}")

    # =========================================================================
    # 2. PROCESSING: FEATURE ATTRIBUTES (Ingest core demographic metrics)
    # =========================================================================
    attr_raw_path = "data/features_attributes.csv"
    df_attr = spark.read.csv(attr_raw_path, header=True, inferSchema=True)\
                  .filter(col('snapshot_date') == snapshot_date)
    print(f"  - Feature Attributes | Extracted payload count: {df_attr.count()}")
    
    attr_dir = os.path.join(bronze_base_directory, "attributes")
    os.makedirs(attr_dir, exist_ok=True)
    attr_filepath = os.path.join(attr_dir, f"bronze_attributes_{suffix}.csv")
    df_attr.toPandas().to_csv(attr_filepath, index=False)
    print(f"  -> Stored trace file: {attr_filepath}")

    # =========================================================================
    # 3. PROCESSING: FEATURE FINANCIALS (Ingest customer ledger summaries)
    # =========================================================================
    fin_raw_path = "data/features_financials.csv"
    df_fin = spark.read.csv(fin_raw_path, header=True, inferSchema=True)\
                  .filter(col('snapshot_date') == snapshot_date)
    print(f"  - Feature Financials | Extracted payload count: {df_fin.count()}")
    
    fin_dir = os.path.join(bronze_base_directory, "financials")
    os.makedirs(fin_dir, exist_ok=True)
    fin_filepath = os.path.join(fin_dir, f"bronze_financials_{suffix}.csv")
    df_fin.toPandas().to_csv(fin_filepath, index=False)
    print(f"  -> Stored trace file: {fin_filepath}")

    # =========================================================================
    # 4. PROCESSING: FEATURE CLICKSTREAM (Ingest continuous event transaction log)
    # =========================================================================
    click_raw_path = "data/feature_clickstream.csv"
    df_click = spark.read.csv(click_raw_path, header=True, inferSchema=True)\
                  .filter(col('snapshot_date') == snapshot_date)
    print(f"  - Feature Clickstream | Extracted payload count: {df_click.count()}")
    
    click_dir = os.path.join(bronze_base_directory, "clickstream")
    os.makedirs(click_dir, exist_ok=True)
    click_filepath = os.path.join(click_dir, f"bronze_clickstream_{suffix}.csv")
    df_click.toPandas().to_csv(click_filepath, index=False)
    print(f"  -> Stored trace file: {click_filepath}")

    print(f"--- [BRONZE TIER] Successfully wrapped ingestion for date: {snapshot_date_str} ---\n")
    return df_lms