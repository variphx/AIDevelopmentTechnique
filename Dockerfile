FROM python:3.12
WORKDIR /app
RUN apt-get update

COPY . .

RUN apt-get install git
RUN git config --global init.defaultBranch main

RUN pip install --no-cache-dir -r requirements.txt
