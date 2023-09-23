FROM ubuntu:22.04


RUN set -xe \
    && apt-get update -y \
    && apt-get install python3-pip -y \
    && apt-get install ufw vim net-tools -y
    
RUN pip install --upgrade pip


LABEL maintainer="sbarathwaj4@gmail.com"

ARG APP_DIR=/usr/src/app

WORKDIR $APP_DIR

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE 6000

ENTRYPOINT ["python3", "./client.py"]