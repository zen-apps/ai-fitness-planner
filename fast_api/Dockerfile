FROM python:3.7-buster

RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get clean

COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

WORKDIR /app
COPY app ./app/

COPY ./gunicorn_config.py ./

EXPOSE 8000

ENV PYTHONUNBUFFERED=True PYTHONPATH=/app/app ENVIRONMENT=prod

CMD exec gunicorn app.main:app -c gunicorn_config.py



