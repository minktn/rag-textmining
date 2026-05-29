FROM python:3.10-slim

WORKDIR /app

COPY workspace/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY workspace/ .

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]