FROM python:3.9-slim

RUN apt-get update && apt-get install -y vim

ENV PYTHONUNBUFFERED 1

ARG APP_DIR=/usr/src/app

WORKDIR $APP_DIR

COPY requirements.txt $APP_DIR/

RUN pip install --no-cache-dir -r requirements.txt

COPY . $APP_DIR/

# Specify the command to run when the container starts
ENTRYPOINT ["python", "fl.kmeans.py"]