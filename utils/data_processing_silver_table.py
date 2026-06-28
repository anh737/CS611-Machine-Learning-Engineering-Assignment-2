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

from pyspark.sql.functions import col, when, lit, regexp_replace, desc, split, abs, size
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, ArrayType

# def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
#     # prepare arguments
#     snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
#     # connect to bronze table
#     partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
#     filepath = bronze_lms_directory + partition_name
#     df = spark.read.csv(filepath, header=True, inferSchema=True)
#     print('loaded from:', filepath, 'row count:', df.count())

#     # clean data: enforce schema / data type
#     # Dictionary specifying columns and their desired datatypes
#     column_type_map = {
#         "loan_id": StringType(),
#         "Customer_ID": StringType(),
#         "loan_start_date": DateType(),
#         "tenure": IntegerType(),
#         "installment_num": IntegerType(),
#         "loan_amt": FloatType(),
#         "due_amt": FloatType(),
#         "paid_amt": FloatType(),
#         "overdue_amt": FloatType(),
#         "balance": FloatType(),
#         "snapshot_date": DateType(),
#     }

#     for column, new_type in column_type_map.items():
#         df = df.withColumn(column, col(column).cast(new_type))

#     # augment data: add month on book
#     df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

#     # augment data: add days past due
#     df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
#     df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
#     df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

#     # save silver table - IRL connect to database to write
#     partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
#     filepath = silver_loan_daily_directory + partition_name
#     df.write.mode("overwrite").parquet(filepath)
#     # df.toPandas().to_parquet(filepath,
#     #           compression='gzip')
#     print('saved to:', filepath)
    
#     return df





def process_silver_table(snapshot_date_str, bronze_base_directory, silver_base_directory, spark):
    """
    Master Silver Tier pipeline to process LMS records alongside Feature Store tables.
    Inputs:
        snapshot_date_str: "YYYY-MM-DD"
        bronze_base_directory: Root landing path for Bronze tier ("datamart/bronze/")
        silver_base_directory: Root output path for Silver tier ("datamart/silver/")
    """
    # Prepare arguments and timestamp objects
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    suffix = snapshot_date_str.replace('-', '_')
    
    print(f"\n--- [SILVER TIER] Executing processing block for date: {snapshot_date_str} ---")

    # =========================================================================
    # 1. TABLE: LMS LOAN DAILY (Label Store Baseline)
    # =========================================================================
    lms_partition_name = "bronze_loan_daily_" + suffix + '.csv'
    lms_filepath = os.path.join(bronze_base_directory, "lms", lms_partition_name)
    df_lms = spark.read.csv(lms_filepath, header=True, inferSchema=True)
    print('Loaded LMS from:', lms_filepath, 'row count:', df_lms.count())

    # Clean data: enforce schema / data type
    lms_column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in lms_column_type_map.items():
        df_lms = df_lms.withColumn(column, col(column).cast(new_type))

    # Augment data: add month on book
    df_lms = df_lms.withColumn("mob", col("installment_num").cast(IntegerType()))

    # Augment data: add days past due
    df_lms = df_lms.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df_lms = df_lms.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df_lms = df_lms.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # Save silver table
    lms_silver_dir = os.path.join(silver_base_directory, "loan_daily")
    os.makedirs(lms_silver_dir, exist_ok=True)
    lms_output_path = os.path.join(lms_silver_dir, "silver_loan_daily_" + suffix + '.parquet')
    df_lms.write.mode("overwrite").parquet(lms_output_path)
    print('Saved LMS to:', lms_output_path)


    # =========================================================================
    # 2. TABLE: FEATURE ATTRIBUTES (Demographic Profiles)
    # =========================================================================
    attr_partition_name = "bronze_attributes_" + suffix + '.csv'
    attr_filepath = os.path.join(bronze_base_directory, "attributes", attr_partition_name)
    df_attr = spark.read.csv(attr_filepath, header=True, inferSchema=True)
    print('Loaded Attributes from:', attr_filepath, 'row count:', df_attr.count())
    
    # ==========================================
    # STEP 1: CLEAN STRING TRASH VALUES
    # ==========================================
    # Replace specific invalid strings with Null (NaN equivalent in Spark)
    df_attr = df_attr.withColumn(
        "Occupation", 
        when(col("Occupation") == "_______", lit(None)).otherwise(col("Occupation"))
    )
    
    df_attr = df_attr.withColumn(
        "SSN", 
        when(col("SSN") == "#F%$D@*&8", lit(None)).otherwise(col("SSN"))
    )
    
    # Remove special characters from Age to recover hidden numbers
    # The regex "[^0-9-]" replaces anything that is NOT a digit (0-9) or a minus sign (-) with an empty string
    df_attr = df_attr.withColumn(
        "Age", 
        regexp_replace(col("Age"), "[^0-9-]", "")
    )
    
    # ==========================================
    # STEP 2: ENFORCE SCHEMA / DATA TYPES
    # ==========================================
    attr_column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(), # Safe to cast now because junk characters are gone
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType()
    }
    
    for column, new_type in attr_column_type_map.items():
        if column in df_attr.columns:
            df_attr = df_attr.withColumn(column, col(column).cast(new_type))
    
    # ==========================================
    # STEP 3: HANDLE OUTLIER AGES WITH MODE
    # ==========================================
    # Define realistic age thresholds
    AGE_THRESHOLD_MIN = 18
    AGE_THRESHOLD_MAX = 100
    
    # Calculate the MODE of Age based strictly on valid data points
    valid_ages = df_attr.filter((col("Age") >= AGE_THRESHOLD_MIN) & (col("Age") <= AGE_THRESHOLD_MAX))
    
    # Check if valid ages exist to prevent errors, then find the most frequent age
    if valid_ages.count() > 0:
        age_mode_row = valid_ages.groupBy("Age").count().orderBy(desc("count")).first()
        age_mode_value = age_mode_row["Age"]
    else:
        age_mode_value = 30 # Fallback default if the entire column is broken
    
    # Replace out-of-bounds ages (and any remaining nulls) with the calculated Mode
    df_attr = df_attr.withColumn(
        "Age",
        when(
            (col("Age") < AGE_THRESHOLD_MIN) | (col("Age") > AGE_THRESHOLD_MAX) | col("Age").isNull(), 
            lit(age_mode_value)
        ).otherwise(col("Age"))
    )
    
    # ==========================================
    # STEP 4: AUGMENT DATA & SAVE TO SILVER
    # ==========================================
    # Drop duplicates and handle primary key missing vectors
    df_attr = df_attr.dropna(subset=["Customer_ID"]).dropDuplicates(["Customer_ID"])
    
    # Save silver table
    attr_silver_dir = os.path.join(silver_base_directory, "attributes")
    os.makedirs(attr_silver_dir, exist_ok=True)
    attr_output_path = os.path.join(attr_silver_dir, "silver_attributes_" + suffix + '.parquet')
    
    df_attr.write.mode("overwrite").parquet(attr_output_path)
    print('Saved Attributes to:', attr_output_path)


    # =========================================================================
    # 3. TABLE: FEATURE FINANCIALS (Customer Balance Records)
    # =========================================================================
    fin_partition_name = "bronze_financials_" + suffix + '.csv'
    fin_filepath = os.path.join(bronze_base_directory, "financials", fin_partition_name)
    df_fin = spark.read.csv(fin_filepath, header=True, inferSchema=True)
    print('Loaded Financials from:', fin_filepath, 'row count:', df_fin.count())
    
    # ==========================================
    # STEP 1: CLEAN CATEGORICAL TRASH VALUES
    # ==========================================
    df_fin = df_fin.withColumn(
        "Payment_Behaviour",
        when(col("Payment_Behaviour") == "!@9#%8", lit(None)).otherwise(col("Payment_Behaviour"))
    )
    
    df_fin = df_fin.withColumn(
        "Credit_Mix",
        when(col("Credit_Mix") == "_", lit(None)).otherwise(col("Credit_Mix"))
    )
    
    df_fin = df_fin.withColumn(
        "Payment_of_Min_Amount",
        when(col("Payment_of_Min_Amount") == "NM", lit(None)).otherwise(col("Payment_of_Min_Amount"))
    )
    
    # ==========================================
    # STEP 2: ENFORCE SCHEMA & CLEAN NUMERIC DATA
    # ==========================================
    fin_column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType()
    }
    
    numeric_cols = [c for c, t in fin_column_type_map.items() if isinstance(t, (FloatType, IntegerType))]
    
    # Define columns that have custom rules for negative/outlier values
    custom_numeric_cols = [
        "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", 
        "Num_of_Loan", "Delay_from_due_date", "Num_Credit_Inquiries"
    ]
    
    for column, new_type in fin_column_type_map.items():
        if column in df_fin.columns:
            if column in numeric_cols:
                # 1. Regex: Remove everything EXCEPT digits (0-9), dot (.), and minus (-)
                cleaned_str = regexp_replace(col(column), "[^0-9\.\-]", "")
                casted_col = cleaned_str.cast(new_type)
                
                if column not in custom_numeric_cols:
                    # Apply general rule: Replace negative values with Null (NaN)
                    final_col = when(casted_col < 0, lit(None)).otherwise(casted_col)
                    df_fin = df_fin.withColumn(column, final_col)
                else:
                    # Just cast it for now, specific logic will be applied in STEP 3
                    df_fin = df_fin.withColumn(column, casted_col)
            else:
                df_fin = df_fin.withColumn(column, col(column).cast(new_type))
    
    # ==========================================
    # STEP 3: CUSTOM BUSINESS LOGIC & OUTLIERS
    # ==========================================
    # Rule 1: Type_of_Loan -> Convert string to array (split by comma and optional space)
    df_fin = df_fin.withColumn("Type_of_Loan", split(col("Type_of_Loan"), ",\s*"))
    
    # Rule 2: Delay_from_due_date -> Convert negative numbers to absolute
    df_fin = df_fin.withColumn("Delay_from_due_date", abs(col("Delay_from_due_date")))
    
    # Define a helper logic to find the mode of valid data within specific thresholds
    def get_mode_value(df, col_name, min_val, max_val):
        valid_df = df.filter((col(col_name) >= min_val) & (col(col_name) <= max_val))
        row = valid_df.groupBy(col_name).count().orderBy(desc("count")).first()
        return row[0] if row else 0
    
    # Calculate Modes
    mode_nba = get_mode_value(df_fin, "Num_Bank_Accounts", 0, 15)
    mode_ncc = get_mode_value(df_fin, "Num_Credit_Card", 0, 15)
    mode_ir = get_mode_value(df_fin, "Interest_Rate", 0, 40)
    mode_nci = get_mode_value(df_fin, "Num_Credit_Inquiries", 0, 25)
    
    # Rule 3, 4, 5: Replace outliers with Mode
    df_fin = df_fin.withColumn(
        "Num_Bank_Accounts",
        when((col("Num_Bank_Accounts") < 0) | (col("Num_Bank_Accounts") > 15), lit(mode_nba))
        .otherwise(col("Num_Bank_Accounts"))
    )
    
    df_fin = df_fin.withColumn(
        "Num_Credit_Card",
        when((col("Num_Credit_Card") < 0) | (col("Num_Credit_Card") > 15), lit(mode_ncc))
        .otherwise(col("Num_Credit_Card"))
    )
    
    df_fin = df_fin.withColumn(
        "Interest_Rate",
        when((col("Interest_Rate") < 0) | (col("Interest_Rate") > 40), lit(mode_ir))
        .otherwise(col("Interest_Rate"))
    )
    
    df_fin = df_fin.withColumn(
        "Num_Credit_Inquiries",
        when((col("Num_Credit_Inquiries") < 0) | (col("Num_Credit_Inquiries") > 25), lit(mode_nci))
        .otherwise(col("Num_Credit_Inquiries"))
    )
    
    # Rule 6: Num_of_Loan -> Replace outliers with the length of Type_of_Loan array
    # Handle cases where Type_of_Loan is Null (size returns -1)
    loan_size = when(col("Type_of_Loan").isNull(), lit(0)).otherwise(size(col("Type_of_Loan")))
    df_fin = df_fin.withColumn(
        "Num_of_Loan",
        when((col("Num_of_Loan") < 0) | (col("Num_of_Loan") > 15), loan_size)
        .otherwise(col("Num_of_Loan"))
    )
    
    # ==========================================
    # STEP 4: AUGMENT DATA & SAVE TO SILVER
    # ==========================================
    # Drop rows without Customer_ID and handle duplicates
    df_fin = df_fin.dropna(subset=["Customer_ID"]).dropDuplicates(["Customer_ID"])
    
    # Save silver table
    fin_silver_dir = os.path.join(silver_base_directory, "financials")
    os.makedirs(fin_silver_dir, exist_ok=True)
    fin_output_path = os.path.join(fin_silver_dir, "silver_financials_" + suffix + '.parquet')
    
    df_fin.write.mode("overwrite").parquet(fin_output_path)
    print('Saved Financials to:', fin_output_path)


    # =========================================================================
    # 4. TABLE: FEATURE CLICKSTREAM (Behavioral App/Website Logs)
    # =========================================================================
    click_partition_name = "bronze_clickstream_" + suffix + '.csv'
    click_filepath = os.path.join(bronze_base_directory, "clickstream", click_partition_name)
    df_click = spark.read.csv(click_filepath, header=True, inferSchema=True)
    print('Loaded Clickstream from:', click_filepath, 'row count:', df_click.count())

    # Clean data: initialize structural schema mapping dictionary
    click_column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType()
    }

    # Dynamically inject anonymized integer features fe_1 through fe_20 into mapping
    for i in range(1, 21):
        click_column_type_map[f"fe_{i}"] = IntegerType()

    # Enforce schema structure and perform data type transformation
    for column, new_type in click_column_type_map.items():
        if column in df_click.columns:
            df_click = df_click.withColumn(column, col(column).cast(new_type))

    # CRITICAL LEAKAGE GUARD: Eliminate temporal target leakage by filtering out 
    # any behavioral data logs occurring strictly after the application point (snapshot_date)
    df_click = df_click.filter(col("snapshot_date") <= F.lit(snapshot_date))
    
    # Extract numerical column names for handling missing or empty feature metrics safely
    numeric_click_cols = [f"fe_{i}" for i in range(1, 21)]
    
    # Drop records missing primary identifiers and fill empty metrics with 0
    df_click = df_click.dropna(subset=["Customer_ID"]) \
                       .fillna(0, subset=numeric_click_cols)

    # Save silver table to target datamart destination
    click_silver_dir = os.path.join(silver_base_directory, "clickstream")
    os.makedirs(click_silver_dir, exist_ok=True)
    click_output_path = os.path.join(click_silver_dir, "silver_clickstream_" + suffix + '.parquet')
    df_click.write.mode("overwrite").parquet(click_output_path)
    print('Saved Clickstream to:', click_output_path)

    print(f"--- [SILVER TIER] Successfully wrapped processing block for date: {snapshot_date_str} ---\n")
    return df_lms