FROM python:3.9-slim

ENV PYTHONUNBUFFERED 1

# Create a directory for your application
ARG APP_DIR=/usr/src/app

WORKDIR $APP_DIR

# Copy the requirements file into the container
COPY requirements.txt $APP_DIR/

# Install any required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python application code into the container
COPY tf-client.py $APP_DIR/

COPY a007c6_*.csv $APP_DIR/

# Specify the command to run when the container starts
CMD ["python", "tf-client.py"]