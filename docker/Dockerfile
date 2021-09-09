FROM python:3.8-slim-buster

ARG token

RUN echo "Europe/Berlin" > /etc/timezone
RUN DEBIAN_FRONTEND="noninteractive" TZ="Europe/Berlin" apt-get -y install tzdata
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /currence
WORKDIR /currence

ENV STREAMLIT_SERVER_PORT=8888
ENV CLIENT_PORT=39003
ENV MONTEL_TOKEN=$token

ENTRYPOINT ["bash", "docker/entrypoint.sh"]
