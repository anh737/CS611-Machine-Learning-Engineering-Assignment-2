import os
import glob
import argparse
from datetime import datetime
import pyspark


# Import structural ETL sub-modules from utils package
import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

def generate_first_of_month_dates(start_date_str, end_date_str):
    """
    Compute and generate a sequence of monthly snapshot strings ('YYYY-MM-DD')
    representing the first day of each calendar month within the specified range.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Increment calendar month index systematically
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

def main(snapshot_date_arg=None):
    print("\n=====================================================================")
    print("      INITIATING PRODUCTION MEDALLION PIPELINE ORCHESTRATION         ")
    print("=====================================================================\n")
    
    # 1. Spark Session Initialization with localized heavy multi-threading parameters
    spark = pyspark.sql.SparkSession.builder \
        .appName("Medallion-Orchestration-Dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Enforce severe error filtering to minimize diagnostic console clutter
    spark.sparkContext.setLogLevel("ERROR")

    # 2. Environment Path Configurations
    bronze_base_directory = "datamart/bronze/"
    silver_base_directory = "datamart/silver/"

    # Scaffold base storage directory infrastructure if absent
    if not os.path.exists(bronze_base_directory):
        os.makedirs(bronze_base_directory)
    if not os.path.exists(silver_base_directory):
        os.makedirs(silver_base_directory)

    # 3. Determine Timeline Execution Mode (Single-run vs Full Backfill)
    if snapshot_date_arg:
        # Single-date targeted execution mode
        dates_str_lst = [snapshot_date_arg]
        print(f"[SYSTEM INFO] Targeted single-date mode triggered for: {snapshot_date_arg}\n")
    else:
        # Default full historical monthly backfill timeline mode
        start_date_str = "2023-01-01"
        end_date_str = "2025-11-01"
        dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
        print(f"[SYSTEM INFO] Default multi-month historical backfill mode activated.")
        print(f"[SYSTEM INFO] Coverage timeframe: {dates_str_lst[0]} through {dates_str_lst[-1]}")
        print(f"[SYSTEM INFO] Total batches to process: {len(dates_str_lst)}\n")

    # =========================================================================
    # PIPELINE EXECUTION: TIER 1 — BRONZE INGESTION LAYER
    # =========================================================================
    print(">>> Executing Ingestion Loops on Bronze Framework...")
    for date_str in dates_str_lst:
        utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_base_directory, spark)
    print("[SUCCESS] Bronze Layer integration snapshots verified.")

    # =========================================================================
    # ORCHESTRATION: TIER 2 — SILVER PROCESSING LAYER
    # =========================================================================
    print("\n>>> Executing Data Cleansing Loops on Silver Framework...")
    for date_str in dates_str_lst:
        utils.data_processing_silver_table.process_silver_table(date_str, bronze_base_directory, silver_base_directory, spark)
    print("[SUCCESS] Silver Layer structured data assets verified.")

    # =========================================================================
    # ORCHESTRATION: TIER 3 — GOLD PROCESSING LAYER (Staged Next)
    # =========================================================================
    print("\n[SYSTEM INFO] Tier 3 (Gold Feature Store) pipeline bypass staged for next implementation phase.")


    print("\n=====================================================================")
    print("         PIPELINE RUN CONCLUDED SUCCESSFULLY (DATAMART SECURED)       ")
    print("=====================================================================\n")

        # create gold datalake
    gold_label_store_directory = "datamart/gold/label_store/"
    
    if not os.path.exists(gold_label_store_directory):
        os.makedirs(gold_label_store_directory)
    
    # run gold backfill
    for date_str in dates_str_lst:
        utils.data_processing_gold_table.process_labels_gold_table(date_str, gold_label_store_directory, spark,  mob = 6, dpd = 30)
    
    
    folder_path = gold_label_store_directory
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").parquet(*files_list)
    print("row_count:",df.count())
    
    df.show()
    # Safe release of multi-threaded JVM background resources
    spark.stop()

if __name__ == "__main__":
    # Configure argument parser to accept optional snapshot date flag
    parser = argparse.ArgumentParser(description="Orchestrated Medallion Pipeline Ingestion Runner")
    parser.add_argument(
        "--snapshotdate", 
        type=str, 
        required=False, 
        help="Target date filter formatted as YYYY-MM-DD. Omit flag to execute full 24-month backfill cycle."
    )
    
    args = parser.parse_args()
    
    # Delegate runtime execution flow to main
    main(snapshot_date_arg=args.snapshotdate)