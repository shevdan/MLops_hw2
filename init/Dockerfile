FROM python:3.10-slim

WORKDIR /app

RUN pip install pandas minio

COPY upload_data.py .

CMD ["python", "upload_data.py"]
