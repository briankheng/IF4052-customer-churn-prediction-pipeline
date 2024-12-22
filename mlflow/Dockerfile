FROM python:3.11-slim-buster
RUN apt-get update && apt-get upgrade -y && apt-get install -y git
COPY requirements.txt .
RUN pip install -r requirements.txt