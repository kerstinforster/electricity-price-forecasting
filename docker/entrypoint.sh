#!/bin/bash

cd $(dirname "$BASH_SOURCE")/..

nohup python3 client.py & > /currence/nohup.out
streamlit run web_app.py