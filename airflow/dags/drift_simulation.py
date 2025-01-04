import datetime
from email.mime import application
import random
import numpy as np
import pandas as pd
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.docker_operator import DockerOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from docker.types import Mount

import mlflow
from airflow import DAG

INPUT_DATA_DIR = "/opt/airflow/data/input_data"
OUTPUT_DATA_DIR = "/opt/airflow/data/output_data"

SPARK_JOBS_DIR = "/opt/spark/jobs"

DRIFT_CHANCE = 0.3
BIAS = 0 # -1 to 1, 0 is no bias, 1 is all drift change upward

def drift_row(type: str, scale):
    """
    type: 'int' | 'float'
    """
    low_rand = -1 + BIAS
    high_rand = 1 + BIAS
    def drift_row_(x):
        if random.random() < DRIFT_CHANCE:
            if type == 'int':
                x += (int)(random.uniform(low_rand, high_rand) * scale)
            else:
                x += random.uniform(low_rand, high_rand) * scale
                x = round(x, 2)
            x = max(0, x)
        return x

    return drift_row_

def drift_categorical(cat):
    cat = pd.Categorical(cat)
    def drift_categorical_(x):
        if random.random() < DRIFT_CHANCE:
            x = random.choice(list(cat.categories))
        return x

    return drift_categorical_

def simulate_drift():
    data = pd.read_csv(f"{INPUT_DATA_DIR}/telco_customer_churn_initial.csv")

    float_cols = ["MonthlyCharges", "TotalCharges"]
    int_cols = ["tenure"]

    for col in float_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        col_mean = data[col].mean()
        data[col] = data[col].apply(drift_row("float", col_mean*0.1))
            
    for col in int_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].apply(drift_row("int", 10))

    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].apply(drift_categorical(data[col]))


    data.to_csv(f"{INPUT_DATA_DIR}/telco_customer_churn_drift.csv", index=False)

    return "Data drift simulation completed."

dag = DAG(
    dag_id="drift_simulation",
    description="Simulate data drift",
    default_args={"owner": "airflow"},
)

simulate_drift = PythonOperator(
    task_id="simulate_drift",
    python_callable=simulate_drift,
    dag=dag,
)

(
    simulate_drift
)