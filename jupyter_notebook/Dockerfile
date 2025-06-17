FROM jupyter/datascience-notebook:latest

USER root

# Install Graphviz system dependencies
RUN apt-get update && apt-get install -y \
    graphviz \
    graphviz-dev \
    pkg-config \
    gcc \
    g++ \
    net-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
COPY ./requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir /app 

RUN mkdir /app/data_docker 

WORKDIR /app 

# Install net-tools package
RUN apt-get update && apt-get install -y net-tools
# docker exec <container_id> bash -c "kill -9 \$(netstat -tulpn | grep ':7860' | awk '{print \$7}' | cut -d'/' -f1)"

CMD '/bin/bash'