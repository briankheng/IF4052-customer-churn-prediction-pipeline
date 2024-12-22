from datetime import datetime

import pandas as pd
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from airflow import DAG

INPUT_DATA_DIR = "/opt/airflow/data/input_data"
OUTPUT_DATA_DIR = "/opt/airflow/data/output_data"


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

    # mlflow.sklearn.autolog()
    # mlflow.set_tracking_uri("http://mlflow:5000")
    # mlflow.set_experiment("customer_churn_prediction")
    # mlflow.start_run()

    svm_model = SVC(kernel="linear", random_state=42)
    svm_model.fit(X_train, y_train)

    # mlflow.sklearn.log_model(svm_model, "model")

    X_test = pd.read_csv(f"{OUTPUT_DATA_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{OUTPUT_DATA_DIR}/y_test.csv")

    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return f"Model accuracy: {accuracy}"
    # return "Training completed."


# def evaluate_model():
#     X_test = pd.read_csv(f"{OUTPUT_DATA_DIR}/X_test.csv")
#     y_test = pd.read_csv(f"{OUTPUT_DATA_DIR}/y_test.csv")

#     model = mlflow.sklearn.load_model("model")
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)

#     mlflow.log_metric("accuracy", accuracy)
#     return "Model evaluation completed."


default_args = {"owner": "airflow"}

dag = DAG(
    dag_id="customer_churn_prediction_model_retrain",
    description="Retrain the customer churn prediction model",
    default_args=default_args,
    # start_date=days_ago(1),
)

read_data = PythonOperator(task_id="read_data", python_callable=read_data, dag=dag)

preprocess_data = PythonOperator(
    task_id="preprocess_data", python_callable=preprocess_data, dag=dag
)

split_train_test = PythonOperator(
    task_id="split_train_test", python_callable=split_train_test, dag=dag
)

train_model = PythonOperator(
    task_id="train_model", python_callable=train_model, dag=dag
)

# evaluate_model = PythonOperator(
#     task_id="evaluate_model", python_callable=evaluate_model, dag=dag
# )

read_data >> preprocess_data >> split_train_test >> train_model
