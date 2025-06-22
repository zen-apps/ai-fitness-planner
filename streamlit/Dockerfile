FROM python:3.11-slim-buster

EXPOSE 8501

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED "1"

COPY . /usr/src/app