from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os

from model_utils import (
    fetch_dataset_from_minio,
    train_and_upload_model,
    load_model_from_minio
)

DATA_BUCKET = "news-dataset"
DATA_KEY = "train/"
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "models-bucket")

app = FastAPI()

class TrainResponse(BaseModel):
    message: str

class Article(BaseModel):
    text: str

class PredictResponse(BaseModel):
    category: str

@app.post("/train", response_model=TrainResponse)
def train(save_local: bool = Query(True, description="Also save model locally?")):
    try:
        df = fetch_dataset_from_minio(DATA_BUCKET, DATA_KEY)
        train_and_upload_model(df, MODEL_BUCKET, save_local)
        return {"message": "Model trained and uploaded to MinIO."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=PredictResponse)
def predict(article: Article):
    try:
        vectorizer, model = load_model_from_minio(MODEL_BUCKET)
        X_vec = vectorizer.transform([article.text])
        prediction = model.predict(X_vec)[0]
        return {"category": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
