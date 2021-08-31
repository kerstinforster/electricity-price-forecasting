#!/bin/bash

cd $(dirname "$BASH_SOURCE")/..

nohup python3 client.py &
streamlit run web_app.py