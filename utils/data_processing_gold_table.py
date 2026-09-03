"""Gold layer: build the label store and the feature set that feeds the model.

Two things happen here, and the split between them is the whole point:

* The **label** is forward-looking — did this loan ever reach ``dpd`` days past
  due within its first ``mob`` months on book. Looking forward is legitimate
  here, because that is the target being predicted.
* The **features** are anchored at the application date (month on book 0), the
  only point in time a real scoring system would have them. Anchoring this way
  is what keeps future information out of the training set.

An earlier version keyed each gold row on the *latest* snapshot in the
performance window. That had two defects: financials and attributes only exist
at month 0, so joining on a later date returned nulls; and clickstream events
from month 6 leaked information the model would not have at scoring time.
"""

import os

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def load_silver_data(start_date_str: str = "2023-01-01", end_date_str: str = "2025-11-01", spark = None):
    """Read all four silver tables and clip them to a snapshot date range.

    Args:
        start_date_str: Inclusive lower bound, ``YYYY-MM-DD``.
        end_date_str: Inclusive upper bound, ``YYYY-MM-DD``.
        spark: Active Spark session.

    Returns:
        ``(df_lms, df_click, df_attr, df_fin)``.
    """
    silver_root_dir = "datamart/silver"

    # Glob each table so one read picks up every monthly partition.
    lms_pattern = os.path.join(silver_root_dir, "loan_daily", "*.parquet")
    click_pattern = os.path.join(silver_root_dir, "clickstream", "*.parquet")
    attr_pattern = os.path.join(silver_root_dir, "attributes", "*.parquet")
    fin_pattern = os.path.join(silver_root_dir, "financials", "*.parquet")

    df_lms_raw = spark.read.parquet(lms_pattern)
    df_click_raw = spark.read.parquet(click_pattern)
    df_attr_raw = spark.read.parquet(attr_pattern)
    df_fin_raw = spark.read.parquet(fin_pattern)

    print(f"  silver window: {start_date_str} to {end_date_str}")
    df_lms = df_lms_raw.filter((F.col("snapshot_date") >= start_date_str) & (F.col("snapshot_date") <= end_date_str))
    df_click = df_click_raw.filter((F.col("snapshot_date") >= start_date_str) & (F.col("snapshot_date") <= end_date_str))
    df_attr = df_attr_raw.filter((F.col("snapshot_date") >= start_date_str) & (F.col("snapshot_date") <= end_date_str))
    df_fin = df_fin_raw.filter((F.col("snapshot_date") >= start_date_str) & (F.col("snapshot_date") <= end_date_str))


    return df_lms, df_click, df_attr, df_fin


def process_labels_gold_table(
    snapshot_date_str,
    gold_label_store_directory,
    spark,
    mob,
    dpd = 30
):
    """Build one gold partition: labels joined to the model feature set.

    Args:
        snapshot_date_str: Snapshot to build, ``YYYY-MM-DD``.
        gold_label_store_directory: Output path for the gold partitions.
        spark: Active Spark session.
        mob: Performance window in months on book.
        dpd: Days past due that define a "bad" loan.
    """
    df_lms, df_click, df_attr, df_fin = load_silver_data(end_date_str = snapshot_date_str,spark = spark)
    suffix = snapshot_date_str.replace('-', '_')
    # --- Labels look forward over months 1..mob; the row is anchored at month 0 ---
    print(f"[gold] {snapshot_date_str} — labelling {dpd}DPD within {mob}MOB, anchored at application")

    # Month on book and days past due, for every installment row.
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

    # Only loans observed for the full window get a label. That keeps the label
    # "mature" and stops it depending on when the pipeline happens to run.
    df_perf_window = df_lms.filter((col("mob_calculated") >= 1) & (col("mob_calculated") <= mob))
    df_label = df_perf_window.groupBy("Customer_ID", "loan_id").agg(
        F.max("dpd_calculated").alias("max_dpd_ever"),
        F.max("mob_calculated").alias("max_mob_observed")
    )
    df_label = df_label.filter(col("max_mob_observed") >= mob) \
        .withColumn("label", F.when(col("max_dpd_ever") >= dpd, 1).otherwise(0).cast(IntegerType())) \
        .withColumn("label_def", F.lit(str(dpd) + 'dpd_within_' + str(mob) + 'mob').cast(StringType()))

    # Anchor the gold row on the application date (month on book 0), so features
    # join at the moment a real system would have them. A side benefit: the row is
    # identical across partitions, which makes de-duplication in train.py trivial.
    df_application = df_lms.filter(col("installment_num") == 0) \
        .select("Customer_ID", "loan_id", col("snapshot_date").alias("snapshot_date"))

    df_default_labels = df_label.join(df_application, on=["Customer_ID", "loan_id"], how="inner")

    print(f"  labelled loans: {df_default_labels.count()}")

    # --- Derived financial ratios ---
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


    # --- Derived demographic features ---
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

    # --- Credit history age, parsed from its free-text form ---
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
            # Credit history age is taken as of the application date, not the date
            # the pipeline runs, so the gold row is deterministic on re-runs.
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

    # --- Join the four tables ---
    # Join on customer AND snapshot date, so a customer's features can never be
    # picked up from a different month than the one being labelled.
    join_keys = ["Customer_ID", "snapshot_date"]
    df_merged = df_default_labels \
        .join(df_fin_gold, on=join_keys, how="left") \
        .join(df_click, on=join_keys, how="left") \
        .join(df_attr_gold, on=join_keys, how="left")

    # --- Final column selection ---
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
    # --- Write the partition ---
    partition_name = "gold_label_store_" + suffix + '.parquet'
    filepath = os.path.join(gold_label_store_directory, partition_name)

    df_final.write.mode("overwrite").parquet(filepath)


    print(f"  wrote {filepath}")

    return df_final