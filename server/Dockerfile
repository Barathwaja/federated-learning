FROM python:3.9-slim

ENV PYTHONUNBUFFERED 1

ARG APP_DIR=/usr/src/app

WORKDIR $APP_DIR

COPY requirements.txt $APP_DIR/

RUN pip install --no-cache-dir -r requirements.txt

COPY . $APP_DIR/

CMD ["python", "server.py"]