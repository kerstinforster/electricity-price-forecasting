#!/bin/bash
DAEMON="--rm -it"
while getopts ":d" opt; do
  case $opt in
    d)
      # If -d flag is specified, use the password stored in the IAIS_CRED env variable
      echo "Docker container starting in the background!"
      DAEMON="-d"
      ;;
    \?)
      echo "Unexpected option -$OPTARG"
      ;;
  esac
done
docker run -p 8888:8888 -p 39003:39003 $DAEMON --name currence-container currence-container