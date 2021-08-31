#!/bin/bash

cd $(dirname "$BASH_SOURCE")/..

docker build -t currence-container -f docker/Dockerfile .
