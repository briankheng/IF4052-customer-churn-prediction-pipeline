from datetime import datetime

import pandas as pd
from airflow.operators.python_operator import PythonOperator

from airflow import DAG


def read_csv():
    df = pd.read_csv("/opt/airflow/data/telco_customer_churn.csv")
    print(df.head())


default_args = {
    "owner": "airflow",
    "retries": 1,
}

with DAG(
    "customer_churn_prediction_model_retrain",
    default_args=default_args,
) as dag:

    read_csv_task = PythonOperator(task_id="read_csv", python_callable=read_csv)

    read_csv_task
