import os
import time
from minio import Minio
import pandas as pd

time.sleep(5)

endpoint = os.environ["MINIO_ENDPOINT"]
access_key = os.environ["MINIO_ROOT_USER"]
secret_key = os.environ["MINIO_ROOT_PASSWORD"]
bucket_name = os.environ["MINIO_BUCKET"]

# Initialize client
client = Minio(
    endpoint=endpoint,
    access_key=access_key,
    secret_key=secret_key,
    secure=False,
)

if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)
    print(f"Bucket '{bucket_name}' created.")
else:
    print(f"ℹBucket '{bucket_name}' already exists.")

file_path = "/data/bbc_news_train.csv"
object_name = "train/bbc_news_train.csv"

client.fput_object(bucket_name, object_name, file_path)
print(f"Uploaded '{object_name}' to bucket '{bucket_name}'")
