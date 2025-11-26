FROM python:3.11-slim

WORKDIR /app

# install system deps (for catboost build/runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app
# ensure src accessible
ENV PYTHONPATH=/app

# install requirements
COPY requirements-api.txt /app/requirements-api.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements-api.txt

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
