# Dynamiv news system

This project aims to train a simple baseline model for the task, provide API for calling the model, utilize object storage (in this case minio locally) and prepare airflow job for uploading new data to training dataset

## Setup

TO run this project, setup .env file with this:

```
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
```

next, run

```bash docker-compose up --build```

## Object storage

As for object storage, here is used minio.

We save model weights there as well as data. On build initial training data is put to minio. On predict model weights are taken from minio.

## Airflow dag

There is airflow job to put new data for training to object store. Currently it is configured to run manually. It takes all the data available and not present in object storage and puts it there

## Model API

It is possible to call API endpoint to train the model with all the available data under API call:

```bash curl -X POST "http://localhost:8000/train?save_local=true"```

There is a parameter "save_local". With True it also saves model weights locally for debug, with False it is only save in minio

There is an endpoint for prediction:

```bash curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "Apple launches new generation iPhone"}'```

And the result will be returned as json {"category": predicted_category}

## Future work

As expected, model performance is poor, there is a lot of further work with this project: using BERT family model, set up new data scraping, deploying model in cloud infrastructure and setting up publicly available API