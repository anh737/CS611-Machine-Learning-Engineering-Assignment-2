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

from pyspark.sql.window import Window
from pyspark.sql.functions import col, when, lit, regexp_replace, desc, split, abs, size, sum, avg, round, countDistinct, count
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, ArrayType


# def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
#     # prepare arguments
#     snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
#     # connect to silver table
#     partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
#     filepath = silver_loan_daily_directory + partition_name
#     df = spark.read.parquet(filepath)
#     print('loaded from:', filepath, 'row count:', df.count())

#     # get customer at mob
#     df = df.filter(col("mob") == mob)

#     # get label
#     df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
#     df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

#     # select columns to save
#     df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

#     # save gold table - IRL connect to database to write
#     partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
#     filepath = gold_label_store_directory + partition_name
#     df.write.mode("overwrite").parquet(filepath)
#     # df.toPandas().to_parquet(filepath,
#     #           compression='gzip')
#     print('saved to:', filepath)
    
#     return df

def load_silver_data(start_date_str: str = "2023-01-01", end_date_str: str = "2025-11-01", spark = None):
    """
    Scans, reads, and filters all 4 Silver layer tables within a specified date range.
    Uses wildcard patterns to load multiple files in parallel without any loops.
    
    Parameters:
    -----------
    start_date_str : str
        The lower bound of the snapshot date range (format: 'YYYY-MM-DD').
    end_date_str : str
        The upper bound of the snapshot date range (format: 'YYYY-MM-DD').
        
    Returns:
    --------
    tuple of PySpark DataFrames
        (df_lms, df_click, df_attr, df_fin) ready for further data pipeline stages.
    """
    # 1. Define the root directory for Silver layer data
    silver_root_dir = "datamart/silver"

    # 2. Formulate paths using wildcard (*) to scan all underlying Parquet files efficiently
    lms_pattern = os.path.join(silver_root_dir, "loan_daily", "*.parquet")
    click_pattern = os.path.join(silver_root_dir, "clickstream", "*.parquet")
    attr_pattern = os.path.join(silver_root_dir, "attributes", "*.parquet")
    fin_pattern = os.path.join(silver_root_dir, "financials", "*.parquet")

    # 3. Read the complete directory contents into parallelized DataFrames
    print(f"[INFO] Scanning Silver layer paths dynamically...")
    df_lms_raw = spark.read.parquet(lms_pattern)
    df_click_raw = spark.read.parquet(click_pattern)
    df_attr_raw = spark.read.parquet(attr_pattern)
    df_fin_raw = spark.read.parquet(fin_pattern)

    # 4. Filter the time-series dynamic dataframes based on the requested date window
    print(f"[INFO] Applying chronological filter: {start_date_str} to {end_date_str}")
    df_lms = df_lms_raw.filter((F.col("snapshot_date") >= start_date_str) & (F.col("snapshot_date") <= end_date_str))
    df_click = df_click_raw.filter((F.col("snapshot_date") >= start_date_str) & (F.col("snapshot_date") <= end_date_str))
    df_attr = df_attr_raw.filter((F.col("snapshot_date") >= start_date_str) & (F.col("snapshot_date") <= end_date_str))
    df_fin = df_fin_raw.filter((F.col("snapshot_date") >= start_date_str) & (F.col("snapshot_date") <= end_date_str))


    print(f"[SUCCESS] All 4 DataFrames loaded and filtered successfully.")
    return df_lms, df_click, df_attr, df_fin



def process_labels_gold_table(
    snapshot_date_str,
    gold_label_store_directory,
    spark,
    mob,
    dpd = 30
):
    """
    Process Silver layer tables to create Gold layer features and labels.
    Merges financial, clickstream, attributes, and loan daily data.
    """
    
    # # 1. Prepare arguments and file paths
    # snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    # suffix = snapshot_date_str.replace('-', '_')
    
    # fin_output_path = os.path.join(fin_silver_dir, "silver_financials_" + suffix + '.parquet')
    # click_output_path = os.path.join(click_silver_dir, "silver_clickstream_" + suffix + '.parquet')
    # attr_output_path = os.path.join(attr_silver_dir, "silver_attributes_" + suffix + '.parquet')
    # lms_output_path = os.path.join(silver_loan_daily_directory, "silver_loan_daily_" + suffix + '.parquet')

    # # 2. Load Silver DataFrames
    # print(f"[INFO] Loading silver tables for snapshot: {snapshot_date_str}")
    # df_fin = spark.read.parquet(fin_output_path)
    # df_click = spark.read.parquet(click_output_path)
    # df_attr = spark.read.parquet(attr_output_path)
    # df_lms = spark.read.parquet(lms_output_path)
    df_lms, df_click, df_attr, df_fin = load_silver_data(end_date_str = snapshot_date_str,spark = spark)
    suffix = snapshot_date_str.replace('-', '_')
    # =========================================================================
    # 3: LABEL (forward mob1-mob) + ANCHOR FEATURES AT APPLICATION (mob 0)
    # ====Quick fix====
    # Cu: nhan lay snapshot MOI NHAT trong cua so mob1-6 lam khoa join feature.
    #     Nhung financials/attributes chi ton tai o mob0 (ngay nop don) -> sai
    #     ngay -> join ra NULL het; con clickstream bi lay o mob6 = leak tuong lai.
    # Moi: nhan van nhin ve tuong lai (mob1-mob, day la TARGET nen hop le), nhung
    #      DONG gold duoc NEO o ngay nop don (mob0) de feature khop dung thoi diem
    #      du doan -> feature tai chinh/nhan khau het NULL, clickstream het leak.
    # ====Quick fix====
    # =========================================================================
    print(f"[INFO] Generating leakage-safe labels: {dpd}DPD within {mob}MOB, anchored at application...")

    # 1. MOB + DPD cho moi dong lich su trong LMS
    df_lms = df_lms.withColumn("mob_calculated", col("installment_num").cast(IntegerType()))
    df_lms = df_lms.withColumn(
        "installments_missed",
        F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())
    ).fillna({"installments_missed": 0})
    df_lms = df_lms.withColumn(
        "first_missed_date",
        F.when(
            col("installments_missed") > 0,
            F.add_months(col("snapshot_date"), -1 * col("installments_missed"))
        ).cast(DateType())
    )
    df_lms = df_lms.withColumn(
        "dpd_calculated",
        F.when(
            col("overdue_amt") > 0.0,
            F.datediff(col("snapshot_date"), col("first_missed_date"))
        ).otherwise(0).cast(IntegerType())
    )

    # 2. ====Quick fix==== Nhan forward-looking tren cua so mob 1..mob.
    #    Chi giu loan da quan sat DU cua so (max mob >= mob) -> nhan "chin",
    #    on dinh, khong phu thuoc ngay xu ly. ====Quick fix====
    df_perf_window = df_lms.filter((col("mob_calculated") >= 1) & (col("mob_calculated") <= mob))
    df_label = df_perf_window.groupBy("Customer_ID", "loan_id").agg(
        F.max("dpd_calculated").alias("max_dpd_ever"),
        F.max("mob_calculated").alias("max_mob_observed")
    )
    df_label = df_label.filter(col("max_mob_observed") >= mob) \
        .withColumn("label", F.when(col("max_dpd_ever") >= dpd, 1).otherwise(0).cast(IntegerType())) \
        .withColumn("label_def", F.lit(str(dpd) + 'dpd_within_' + str(mob) + 'mob').cast(StringType()))

    # 3. ====Quick fix==== Neo o NGAY NOP DON = snapshot tai mob 0 (loan_start).
    #    snapshot_date cua dong gold = ngay nop don -> feature join dung thoi diem,
    #    va dong gold giong het nhau giua cac partition -> de khu trung o train.
    # ====Quick fix====
    df_application = df_lms.filter(col("installment_num") == 0) \
        .select("Customer_ID", "loan_id", col("snapshot_date").alias("snapshot_date"))

    df_default_labels = df_label.join(df_application, on=["Customer_ID", "loan_id"], how="inner")

    print(f"[SUCCESS] Matured labels anchored at application: {df_default_labels.count()}")
    
    # ---------------------------------------------------------
    # 4. Feature Engineering (Gold Layer) on Financial table
    # ---------------------------------------------------------
    print("[INFO] Engineering Gold features...")
    df_fin_gold = df_fin \
        .withColumn(
            "Debt_to_Income_Ratio",
            F.when(
                (F.col("Annual_Income").isNotNull()) & (F.col("Annual_Income") > 0),
                F.col("Outstanding_Debt") / F.col("Annual_Income")
            ).otherwise(0.0)
        ) \
        .withColumn(
            "EMI_to_Salary_Ratio",
            F.when(
                (F.col("Monthly_Inhand_Salary").isNotNull()) & (F.col("Monthly_Inhand_Salary") > 0),
                F.col("Total_EMI_per_month") / F.col("Monthly_Inhand_Salary")
            ).otherwise(0.0)
        ) \
        .withColumn(
            "Savings_Propensity",
            F.when(
                (F.col("Monthly_Inhand_Salary").isNotNull()) & (F.col("Monthly_Inhand_Salary") > 0),
                (F.coalesce(F.col("Monthly_Balance"), F.lit(0.0)) + F.coalesce(F.col("Amount_invested_monthly"), F.lit(0.0))) / F.col("Monthly_Inhand_Salary")
            ).otherwise(0.0)
        ) \
        .withColumn(
            "Spend_Level_Idx",
            F.when(F.col("Payment_Behaviour").contains("Low_spent"), 33.39)
             .when(F.col("Payment_Behaviour").contains("High_spent"), 23.47)
             .otherwise(28.96).cast(IntegerType()) # Default to 1 (Low) if UNKNOWN shares the same risk profile for Logistic Regression
        ) \
        .withColumn(
            "Payment_Value_Level_Idx",
            F.when(F.col("Payment_Behaviour").contains("Small_value"), 33.59)
             .when(F.col("Payment_Behaviour").contains("Medium_value"), 27.11)
             .when(F.col("Payment_Behaviour").contains("Large_value"), 23.67)
             .otherwise(28.96).cast(IntegerType()) # Default to 2 (Medium) as a neutral imputation for Logistic Regression
        )


    # ---------------------------------------------------------
    # 5. Feature Engineering (Gold Layer) on Attribute table
    # ---------------------------------------------------------
    df_attr_gold = df_attr \
        .withColumn(
            "Is_Age_gt_45",
            F.when(F.col("Age") >= 45, 1).otherwise(0)
        ) \
        .withColumn(
        "Occupation_Idx",
        F.when(F.col("Occupation") == "Accountant", 30.59)
         .when(F.col("Occupation") == "Writer", 30.49)
         .when(F.col("Occupation") == "Mechanic", 30.13)
         .when(F.col("Occupation") == "Teacher", 29.67)
         .when(F.col("Occupation") == "Engineer", 29.51)
         .when(F.col("Occupation") == "Manager", 29.48)
         .when(F.col("Occupation") == "Entrepreneur", 29.25)
         .when(F.col("Occupation") == "Developer", 29.10)
         .when(F.col("Occupation") == "Journalist", 29.04)
         .when(F.col("Occupation") == "Scientist", 28.64)
         .when(F.col("Occupation") == "Doctor", 28.55)
         .when(F.col("Occupation") == "Musician", 28.21)
         .when(F.col("Occupation") == "Architect", 27.30)
         .when(F.col("Occupation") == "Lawyer", 26.57)
         .when(F.col("Occupation") == "Media_Manager", 26.28)
         # Handle UNKNOWN or any missing/new occupations using the baseline UNKNOWN rate (28.52)
         .otherwise(28.52).cast(FloatType())
    )

    # ---------------------------------------------------------
    # DYNAMIC TIME-LAPSE CALCULATION FOR CREDIT HISTORY AGE
    # ---------------------------------------------------------
    df_fin_gold = df_fin_gold \
        .withColumn(
            "Extracted_Years", 
            F.coalesce(F.regexp_extract(F.col("Credit_History_Age"), "(\\d+)\\s*Year", 1).cast("int"), F.lit(0))
        ) \
        .withColumn(
            "Extracted_Months", 
            F.coalesce(F.regexp_extract(F.col("Credit_History_Age"), "(\\d+)\\s*Month", 1).cast("int"), F.lit(0))
        ) \
        .withColumn(
            "Base_History_Months", 
            (F.col("Extracted_Years") * 12) + F.col("Extracted_Months")
        ) \
        .withColumn(
            # ====Quick fix==== Tinh tuoi lich su tin dung AS OF ngay nop don (snapshot_date)
            # thay vi theo snapshot_date_str (ngay xu ly) -> bo phu thuoc thoi diem chay
            # pipeline, dong gold deterministic + khong leak du lieu tuong lai. ====Quick fix====
            "Months_From_Input_To_Snapshot",
            F.lit(0)
        ) \
        .withColumn(
            "Credit_History_Age_Months",
            F.round(F.col("Base_History_Months") + F.col("Months_From_Input_To_Snapshot"), 2)
        ) \
        .withColumn(
            "Credit_History_Age_Years",
            F.round(F.col("Credit_History_Age_Months") / 12, 0).cast("int")
        ) \
        .withColumn(
            "is_Credit_Age_gt_15",
            F.when(F.col("Credit_History_Age_Years") > 15, 1).otherwise(0)
        ) \
        .drop("Extracted_Years", "Extracted_Months", "Base_History_Months", "Months_From_Input_To_Snapshot")

    # =========================================================================
    # 6. Merge all 4 tables (DUAL-KEY POLICY)
    # =========================================================================
    print("[INFO] Merging all DataFrames using a leakage-proof dual-key constraint...")
    
    # Define explicit compound join keys to eliminate cross-date leakage
    join_keys = ["Customer_ID", "snapshot_date"]
    
    # Perform left joins matching both customer and the exact snapshot timeline
    df_merged = df_default_labels \
        .join(df_fin_gold, on=join_keys, how="left") \
        .join(df_click, on=join_keys, how="left") \
        .join(df_attr_gold, on=join_keys, how="left")

    # ---------------------------------------------------------
    # 7. Select desired columns
    # ---------------------------------------------------------
    # Selecting the final columns as per instruction
    df_final = df_merged.select(
        "loan_id", 
        "Customer_ID",
        "label",
        "label_def", 
        "snapshot_date",
        "Debt_to_Income_Ratio", 
        "EMI_to_Salary_Ratio", 
        "Savings_Propensity", 
        "Credit_History_Age_Years",
        "is_Credit_Age_gt_15",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Num_of_Loan",
        "Interest_Rate",
        "Monthly_Inhand_Salary",
        "Annual_Income",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
        "Is_Age_gt_45",
        "Age",
        "Occupation",
        "Occupation_Idx",
        "Payment_Behaviour",
        "Spend_Level_Idx",
        "Payment_Value_Level_Idx",
        "fe_1",
        "fe_2",
        "fe_3",
        "fe_4",
        "fe_5",
        "fe_6",
        "fe_7",
        "fe_8",
        "fe_9",
        "fe_10",
        "fe_11",
        "fe_12",
        "fe_13",
        "fe_14",
        "fe_15",
        "fe_16",
        "fe_17",
        "fe_18",
        "fe_19",
        "fe_20"
    )
    # ---------------------------------------------------------
    # 8. Save Gold Table
    # ---------------------------------------------------------
    partition_name = "gold_label_store_" + suffix + '.parquet'
    filepath = os.path.join(gold_label_store_directory, partition_name)
    
    print(f"[INFO] Writing Gold table to {filepath}...")
    df_final.write.mode("overwrite").parquet(filepath)
    # df_final.toPandas().to_parquet(filepath, compression='gzip')

    
    print('[SUCCESS] Saved to:', filepath)
    
    return df_final