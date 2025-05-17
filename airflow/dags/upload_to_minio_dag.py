from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta
from minio import Minio
import os
from pathlib import Path

from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
BUCKET_NAME = "news-dataset"
UPLOAD_PREFIX = "train/" 

DATA_DIR = "/opt/airflow/data"

# Python task
def upload_missing_files():
    client = Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)

    for file_path in Path(DATA_DIR).glob("batch_*.csv"):
        file_name = file_path.name
        remote_key = f"{UPLOAD_PREFIX}{file_name}"

        try:
            client.stat_object(BUCKET_NAME, remote_key)
            print(f"⏭️ Skipping: {remote_key} already exists.")
        except Exception:
            client.fput_object(
                BUCKET_NAME,
                remote_key,
                str(file_path),
                content_type="application/csv",
            )
            print(f"Uploaded: {file_path} to s3://{BUCKET_NAME}/{remote_key}")

with DAG(
    dag_id="upload_daily_to_minio",
    default_args=default_args,
    description="Upload any new batch_*.csv files to MinIO for training",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["minio", "upload", "mlops"],
) as dag:

    upload_task = PythonOperator(
        task_id="upload_missing_files",
        python_callable=upload_missing_files,
    )
