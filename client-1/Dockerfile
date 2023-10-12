FROM python:3.9-slim

RUN apt-get update && apt-get install -y vim

ENV PYTHONUNBUFFERED 1

ARG APP_DIR=/usr/src/app

WORKDIR $APP_DIR

COPY requirements.txt $APP_DIR/

RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python application code into the container
# COPY geo.lstm.py $APP_DIR/
COPY geo.kmeans.py $APP_DIR/

COPY data/geoaltitude/*.csv $APP_DIR/data/geoaltitude/

# Specify the command to run when the container starts
# CMD ["python", "geo.lstm.py"]