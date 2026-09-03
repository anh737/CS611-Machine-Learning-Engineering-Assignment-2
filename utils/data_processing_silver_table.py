"""Silver layer: enforce schema, repair known junk values, and de-duplicate.

The raw feed carries placeholder garbage (``_______`` occupations, ``#F%$D@*&8``
SSNs, negative account counts, ages in the hundreds). Everything here is about
making those columns trustworthy before any feature is derived from them.
"""

import os
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, lit, regexp_replace, desc, split, abs, size
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table(snapshot_date_str, bronze_base_directory, silver_base_directory, spark):
    """Clean and type one monthly snapshot of all four bronze tables.

    Args:
        snapshot_date_str: Snapshot to process, ``YYYY-MM-DD``.
        bronze_base_directory: Root bronze path, e.g. ``datamart/bronze/``.
        silver_base_directory: Root silver path, e.g. ``datamart/silver/``.
        spark: Active Spark session.
    """

    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    suffix = snapshot_date_str.replace('-', '_')

    print(f"[silver] {snapshot_date_str}")

    # --- 1. LMS loan daily — basis for the labels ---
    lms_partition_name = "bronze_loan_daily_" + suffix + '.csv'
    lms_filepath = os.path.join(bronze_base_directory, "lms", lms_partition_name)
    df_lms = spark.read.csv(lms_filepath, header=True, inferSchema=True)
    print(f"  lms: {df_lms.count()} rows")

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


    # --- 2. Feature attributes — customer demographics ---
    attr_partition_name = "bronze_attributes_" + suffix + '.csv'
    attr_filepath = os.path.join(bronze_base_directory, "attributes", attr_partition_name)
    df_attr = spark.read.csv(attr_filepath, header=True, inferSchema=True)
    print(f"  attributes: {df_attr.count()} rows")

    # Replace the feed's placeholder strings with null.
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

    # Enforce the schema.
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

    # Ages outside 18-100 are data errors; fall back to the modal valid age.
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

    # Drop rows with no primary key, then de-duplicate.
    # Drop duplicates and handle primary key missing vectors
    df_attr = df_attr.dropna(subset=["Customer_ID"]).dropDuplicates(["Customer_ID"])

    # Save silver table
    attr_silver_dir = os.path.join(silver_base_directory, "attributes")
    os.makedirs(attr_silver_dir, exist_ok=True)
    attr_output_path = os.path.join(attr_silver_dir, "silver_attributes_" + suffix + '.parquet')

    df_attr.write.mode("overwrite").parquet(attr_output_path)


    # --- 3. Feature financials — income, debt, repayment behaviour ---
    fin_partition_name = "bronze_financials_" + suffix + '.csv'
    fin_filepath = os.path.join(bronze_base_directory, "financials", fin_partition_name)
    df_fin = spark.read.csv(fin_filepath, header=True, inferSchema=True)
    print(f"  financials: {df_fin.count()} rows")

    # Replace the feed's placeholder strings with null.
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

    # Enforce the schema and strip stray characters out of numeric columns.
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
                    # Cast only; the column-specific repair rules run below.
                    df_fin = df_fin.withColumn(column, casted_col)
            else:
                df_fin = df_fin.withColumn(column, col(column).cast(new_type))

    # Column-specific repair rules.
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

    # Drop rows with no primary key, then de-duplicate.
    # Drop rows without Customer_ID and handle duplicates
    df_fin = df_fin.dropna(subset=["Customer_ID"]).dropDuplicates(["Customer_ID"])

    # Save silver table
    fin_silver_dir = os.path.join(silver_base_directory, "financials")
    os.makedirs(fin_silver_dir, exist_ok=True)
    fin_output_path = os.path.join(fin_silver_dir, "silver_financials_" + suffix + '.parquet')

    df_fin.write.mode("overwrite").parquet(fin_output_path)


    # --- 4. Feature clickstream — anonymised behavioural counters ---
    click_partition_name = "bronze_clickstream_" + suffix + '.csv'
    click_filepath = os.path.join(bronze_base_directory, "clickstream", click_partition_name)
    df_click = spark.read.csv(click_filepath, header=True, inferSchema=True)
    print(f"  clickstream: {df_click.count()} rows")

    # Schema map for the clickstream table.
    click_column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType()
    }

    # fe_1 .. fe_20 are the anonymised behavioural counters.
    for i in range(1, 21):
        click_column_type_map[f"fe_{i}"] = IntegerType()

    # Enforce the schema.
    for column, new_type in click_column_type_map.items():
        if column in df_click.columns:
            df_click = df_click.withColumn(column, col(column).cast(new_type))

    # Leakage guard: drop behavioural events logged after the application date.
    df_click = df_click.filter(col("snapshot_date") <= F.lit(snapshot_date))


    numeric_click_cols = [f"fe_{i}" for i in range(1, 21)]

    # Drop rows with no primary key; a missing counter means no events, so fill 0.
    df_click = df_click.dropna(subset=["Customer_ID"]) \
                       .fillna(0, subset=numeric_click_cols)


    click_silver_dir = os.path.join(silver_base_directory, "clickstream")
    os.makedirs(click_silver_dir, exist_ok=True)
    click_output_path = os.path.join(click_silver_dir, "silver_clickstream_" + suffix + '.parquet')
    df_click.write.mode("overwrite").parquet(click_output_path)


    print(f"  done: {snapshot_date_str}")
    return df_lms