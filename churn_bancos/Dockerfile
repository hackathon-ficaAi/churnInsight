FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline_churn_rf.joblib .
COPY app.py .
COPY schema_pipeline.json .


EXPOSE 8000

# CMD ["uvicorn", "app:app","--reload", "--host", "0.0.0.0", "--port", "8000"]