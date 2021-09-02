FROM ubuntu:20.04

ARG token

RUN apt-get update && apt-get install -y python3.8 python3-pip python3.8-venv git

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /currence
WORKDIR /currence

ENV STREAMLIT_SERVER_PORT=8888
ENV CLIENT_PORT=39003
ENV MONTEL_TOKEN=$token

ENTRYPOINT ["bash", "docker/entrypoint.sh"]