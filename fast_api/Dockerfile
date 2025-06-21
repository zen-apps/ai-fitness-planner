FROM python:3.11-slim-buster

RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get clean

WORKDIR /app
COPY ./requirements.txt ./requirements.txt
COPY app ./app/

RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt && rm -rf pip3 cache dir

EXPOSE 8000

#ENV PYTHONUNBUFFERED=True PYTHONPATH=/app/app

ENTRYPOINT ["uvicorn"]
