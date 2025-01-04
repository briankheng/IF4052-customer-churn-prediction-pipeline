# Customer Churn Prediction Pipeline

This project aims to predict customer churn using a machine learning pipeline.

## Prerequisites

Ensure you have Docker and Docker Compose installed on your machine.

## How to Run

To run the project, use Docker Compose:

```sh
docker compose up --build
```

## Accessing Airflow

Once the services are up, you can access the Airflow web interface at:

[http://localhost:8080/](http://localhost:8080/)

**Credentials:**
- Username: `airflow`
- Password: `airflow`

### Workflows

#### customer_churn_prediction_model_retrain
Before running, make sure:
- there is bucket named 'mlflow' in MinIO [http://localhost:9001/buckets](http://localhost:9001/buckets)
- there is connection 'spark_local' pointing to spark://spark:7077 in airflow


