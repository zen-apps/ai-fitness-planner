FROM python:3.11-slim-buster

RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get clean

COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt && rm -rf pip3 cache dir

WORKDIR /app
COPY app ./app/

EXPOSE 8000

#ENV PYTHONUNBUFFERED=True PYTHONPATH=/app/app

ENTRYPOINT ["uvicorn"]
