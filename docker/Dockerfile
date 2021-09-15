FROM python:3.8-slim-buster

ARG token

RUN echo "Europe/Berlin" > /etc/timezone
RUN DEBIAN_FRONTEND="noninteractive" TZ="Europe/Berlin" apt-get -y install tzdata
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY . /currence
WORKDIR /currence

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV STREAMLIT_SERVER_PORT=8888
ENV CLIENT_PORT=39003
ENV MONTEL_TOKEN=$token

ENTRYPOINT ["bash", "docker/entrypoint.sh"]
