import os
import joblib
import pandas as pd
from minio import Minio
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_OBJ_KEY = "news-models/latest/model.pkl"

def get_minio_client():
    return Minio(
        endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        secure=False
    )

def fetch_dataset_from_minio(bucket: str, prefix: str = "train/") -> pd.DataFrame:
    client = get_minio_client()
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)
    print(objects)
    dfs = []
    for obj in objects:
        if obj.object_name.endswith(".csv"):
            print(f"📦 Reading {obj.object_name}")
            response = client.get_object(bucket, obj.object_name)
            df = pd.read_csv(BytesIO(response.read()))
            dfs.append(df)

    if not dfs:
        raise ValueError("No CSV files found in training bucket.")

    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df


def train_and_upload_model(df: pd.DataFrame, model_bucket: str, save_local: bool = True):
    X = df["Text"]
    y = df["Category"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    bytes_io = BytesIO()
    joblib.dump((vectorizer, model), bytes_io)
    bytes_io.seek(0)

    client = get_minio_client()

    if not client.bucket_exists(model_bucket):
        client.make_bucket(model_bucket)

    client.put_object(
        model_bucket,
        MODEL_OBJ_KEY,
        data=bytes_io,
        length=bytes_io.getbuffer().nbytes,
        content_type="application/octet-stream"
    )

    if save_local:
        os.makedirs("trained_models", exist_ok=True)
        joblib.dump((vectorizer, model), "trained_models/model.pkl")
        print("Local model saved to trained_models/model.pkl")

def load_model_from_minio(model_bucket: str):
    client = get_minio_client()
    response = client.get_object(model_bucket, MODEL_OBJ_KEY)
    return joblib.load(BytesIO(response.read()))
