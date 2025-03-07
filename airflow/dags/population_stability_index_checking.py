import datetime
from email.mime import application
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

import customer_churn_prediction_model_retrain

import mlflow
from airflow import DAG

INPUT_DATA_DIR = "/opt/airflow/data/input_data"
OUTPUT_DATA_DIR = "/opt/airflow/data/output_data"

SPARK_JOBS_DIR = "/opt/spark/jobs"

def read_data_drift():
    data = pd.read_csv(f"{INPUT_DATA_DIR}/telco_customer_churn_drift.csv")
    data.to_csv(f"{OUTPUT_DATA_DIR}/telco_customer_churn_drift.csv", index=False)

    return "Reading data completed."

def split_train_test_drift():
    data = pd.read_csv(f"{OUTPUT_DATA_DIR}/preprocessed_telco_customer_churn_drift.csv")
    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train.to_csv(f"{OUTPUT_DATA_DIR}/X_train_drift.csv", index=False)
    X_test.to_csv(f"{OUTPUT_DATA_DIR}/X_test_drift.csv", index=False)
    y_train.to_csv(f"{OUTPUT_DATA_DIR}/y_train_drift.csv", index=False)
    y_test.to_csv(f"{OUTPUT_DATA_DIR}/y_test_drift.csv", index=False)

    return "Splitting data into training and test sets completed."

def count_psi(X_test, X_test_drift, num_bins=10):
    model_name="customer_churn_prediction_model"
    model_version="latest"
    
    model_uri=f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict_proba(X_test)[:,0]
    y_pred_drift = model.predict_proba(X_test_drift)[:,0]

    eps = 1e-4

    y_pred.sort()
    y_pred_drift.sort()

    min_val = min(min(y_pred), min(y_pred_drift))
    max_val = max(max(y_pred), max(y_pred_drift))

    bins = [min_val + (max_val-min_val)*(i)/num_bins for i in range(num_bins+1)]
    bins[0] = min_val-eps
    bins[-1] = max_val+eps

    bins_init = pd.cut(y_pred, bins=bins, labels=range(1, num_bins+1))
    df_init = pd.DataFrame({'init': y_pred, 'bin': bins_init})
    grp_init = df_init.groupby('bin').count()
    grp_init['percent_init'] = grp_init['init'] / sum(grp_init['init'])

    bins_drift = pd.cut(y_pred_drift, bins=bins, labels=range(1, num_bins+1))
    df_drift = pd.DataFrame({'drift': y_pred_drift, 'bin': bins_drift})
    grp_drift = df_drift.groupby('bin').count()
    grp_drift['percent_drift'] = grp_drift['drift'] / sum(grp_drift['drift'])


    psi_df = grp_init.join(grp_drift, how="inner",on="bin")
    psi_df['percent_init'] = psi_df['percent_init'].apply(lambda x: eps if x == 0 else x)
    psi_df['percent_drift'] = psi_df['percent_drift'].apply(lambda x: eps if x == 0 else x)

    psi_df['psi'] = (psi_df['percent_init'] - psi_df['percent_drift']) * np.log(psi_df['percent_init'] / psi_df['percent_drift'])

    print(psi_df)
    print(psi_df.sum())

    return psi_df['psi'].sum()

def check_drift():
    X_test = pd.read_csv(f"{OUTPUT_DATA_DIR}/X_test.csv")
    X_test_drift = pd.read_csv(f"{OUTPUT_DATA_DIR}/X_test_drift.csv")

    psi = count_psi(X_test, X_test_drift)

    print("PSI: ", psi)

    return psi > 0.1

def change_data():
    drift_data = pd.read_csv(f"{INPUT_DATA_DIR}/telco_customer_churn_drift.csv")
    drift_data.to_csv(f"{INPUT_DATA_DIR}/telco_customer_churn.csv", index=False)


default_args = {"owner": "airflow"}
dag = DAG(
    dag_id="population_stability_index_checking",
    description="Check model drift",
    default_args=default_args,
    schedule_interval="0 7 * * *",
    start_date=datetime.datetime.now()-datetime.timedelta(days=1),
)

read_data_drift = PythonOperator(
    task_id="read_data_drift",
    python_callable=read_data_drift,
    dag=dag,
)

preprocess_data_spark_drift = SparkSubmitOperator(
    application=f"{SPARK_JOBS_DIR}/preprocess_data_drift.py",
    task_id="preprocess_data_spark_drift",
    name="preprocess_data_spark_drift",
    conn_id="spark_local",
    verbose=True,
    dag=dag,
)

split_train_test_drift = PythonOperator(
    task_id="split_train_test_drift",
    python_callable=split_train_test_drift,
    dag=dag,
)

check_drift = ShortCircuitOperator(
    task_id="check_drift",
    python_callable=check_drift,
    dag=dag,
)

change_data = PythonOperator(
    task_id="change_data",
    python_callable=change_data,
    dag=dag,
)






# Taken from customer_churn_prediction_model_retrain.py
# Yes, Do Repeat Myself! (sorry)

def read_data():
    data = pd.read_csv(f"{INPUT_DATA_DIR}/telco_customer_churn.csv")
    data.to_csv(f"{OUTPUT_DATA_DIR}/telco_customer_churn.csv", index=False)

    return "Reading data completed."


def preprocess_data():
    data = pd.read_csv(f"{OUTPUT_DATA_DIR}/telco_customer_churn.csv")
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data.fillna(data["TotalCharges"].mean(), inplace=True)
    data = data.drop(columns=["customerID"])

    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = pd.Categorical(data[column])
        data[column] = data[column].cat.codes

    data.to_csv(f"{OUTPUT_DATA_DIR}/preprocessed_telco_customer_churn.csv", index=False)

    return "Preprocessing data completed."


def split_train_test():
    data = pd.read_csv(f"{OUTPUT_DATA_DIR}/preprocessed_telco_customer_churn.csv")
    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train.to_csv(f"{OUTPUT_DATA_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{OUTPUT_DATA_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{OUTPUT_DATA_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUTPUT_DATA_DIR}/y_test.csv", index=False)

    return "Splitting data into training and test sets completed."


def train_model():
    X_train = pd.read_csv(f"{OUTPUT_DATA_DIR}/X_train.csv")
    y_train = pd.read_csv(f"{OUTPUT_DATA_DIR}/y_train.csv")

    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        svm_model = SVC(kernel="linear", probability=True, random_state=42, max_iter=10)
        svm_model.fit(X_train, y_train)

        run_id = run.info.run_id

    return run_id


def register_model(mlflow_run_id):
    X_test = pd.read_csv(f"{OUTPUT_DATA_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{OUTPUT_DATA_DIR}/y_test.csv")

    model_uri = f"runs:/{mlflow_run_id}/model"

    model = mlflow.pyfunc.load_model(model_uri)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metric("test_accuracy", float(accuracy))

    mlflow.register_model(model_uri, "customer_churn_prediction_model")

    return "Model registered."

read_data = PythonOperator(task_id="read_data", python_callable=read_data, dag=dag)

preprocess_data_spark = SparkSubmitOperator(
    application=f"{SPARK_JOBS_DIR}/preprocess_data.py",
    task_id="preprocess_data_spark",
    name="preprocess_data_spark",
    conn_id="spark_local",
    verbose=True,
    dag=dag,
)

split_train_test = PythonOperator(
    task_id="split_train_test", python_callable=split_train_test, dag=dag
)

train_model = PythonOperator(
    task_id="train_model", python_callable=train_model, dag=dag
)

register_model = PythonOperator(
    task_id="register_model",
    python_callable=register_model,
    op_args=["{{ ti.xcom_pull(task_ids='train_model') }}"],
    dag=dag,
)

restart_prediction_service = BashOperator(
    task_id="restart_prediction_service",
    bash_command="docker restart prediction_service",
    dag=dag,
)

(
    read_data
    >> preprocess_data_spark
    >> split_train_test
    >> train_model
    >> register_model
    >> restart_prediction_service
)



(
    read_data_drift
    >> preprocess_data_spark_drift
    >> split_train_test_drift
    >> check_drift 
    >> change_data 
)

(
    change_data
    >> read_data
)