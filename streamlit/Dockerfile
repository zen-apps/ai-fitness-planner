FROM python:3.11-slim-buster

EXPOSE 8501

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED "1"

# Copy and set environment variables
COPY config/dev.env ./
RUN export $(cat dev.env | xargs)

COPY . /usr/src/app