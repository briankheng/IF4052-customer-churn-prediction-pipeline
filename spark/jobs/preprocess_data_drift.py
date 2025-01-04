import pandas as pd
from pyspark.sql import SparkSession
from pyspark import pandas as ps
from pyspark.sql.functions import col, mean, dense_rank
from pyspark.sql.window import Window
import logging

spark = SparkSession.builder.master("local[*]").appName("TelcoChurn").getOrCreate()

INPUT_DATA_DIR = "/opt/airflow/data/input_data"
OUTPUT_DATA_DIR = "/opt/airflow/data/output_data"

# data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
#     data.fillna(data["TotalCharges"].mean(), inplace=True)
#     data = data.drop(columns=["customerID"])

#     for column in data.select_dtypes(include=["object"]).columns:
#         data[column] = pd.Categorical(data[column])
#         data[column] = data[column].cat.codes



def preprocess_data_spark():
    data = spark.read.csv(f"{OUTPUT_DATA_DIR}/telco_customer_churn_drift.csv", header=True, inferSchema=True)

    # Handle missing and invalid data
    data = data.withColumn("TotalCharges", col("TotalCharges").cast("int"))
    avg_total_charges = data.select(mean(col("TotalCharges"))).collect()[0][0]
    data = data.fillna({"TotalCharges": avg_total_charges}).drop("customerID")

    # data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    # data.fillna(data["TotalCharges"].mean(), inplace=True)
    # data = data.drop(columns=["customerID"])

    # Encode categorical columns
    for column in [col for col in data.columns if data.schema[col].dataType.simpleString() == 'string']:
        data = data.withColumn(column, dense_rank().over(Window.orderBy(column)))
        


    data.toPandas().to_csv(f"{OUTPUT_DATA_DIR}/preprocessed_telco_customer_churn_drift.csv", index=False)
    return "Preprocessing data with Spark completed."

if __name__ == "__main__":
    preprocess_data_spark()