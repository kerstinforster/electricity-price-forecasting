![Currence Logo](src/frontend/resources/currence_logo_big.png)

# Currence Project

This is the project repository of group 3 for the course Applied Machine Intelligence at TUM.  
Group Members:
- Adrian Michl (MTK: 03672576)
- Anna Mrosik (MTK: 03695133)
- Florian Mysliwetz (MTK: 03671374)
- Jakob Kruse (MTK: 03682020)
- Kerstin Forster (MTK: 03696804)
- Oliver Hehmann (MTK: 03739107)
- Philip Durst (MTK: 03684092)

## Setup
1. Install the required packages:
```console
sudo apt-get install python3.8 python3.8-venv
```
2. Create a new Python 3.8 virtual environment:
```console
python3.8 -m venv venv
source venv/bin/activate
``` 
3. Install the required python packages:
```console
pip install -r requirements.txt
```

## Frontend
Run the frontend / web application:
Go to the root repository 'group03' in your terminal and then type:
```console
streamlit run web_app.py
```

## Docker
The docker container of this repository already has all dependencies installed and runs
out of the box.   
You can either build the container yourself or download the prebuilt container.  
To build the container, run:
```console
docker build -t currence-container -f docker/Dockerfile --build-arg token=TOKEN .
```
Hereby, the `TOKEN` build argument is optional and should be replaced with the current Montel API Bearer Token.  

To download the prebuild container, run:
```console
docker pull gitlab.ldv.ei.tum.de:5005/ami2021/group03
docker tag gitlab.ldv.ei.tum.de:5005/ami2021/group03 currence-container
```

Finally, run the container using:
```console
bash docker/docker_run.sh
```
You can add an optional `-d` flag to the docker run command to run the docker container in the background.
In order to update the token in the running container, use the following command in a new terminal:
```console
docker exec currence-container python3 docker/set_token.py TOKEN
```
## Linter
Run the linter:
```console
pylint src test
```

## Tests
Run the unit tests:
```console
python -m pytest
```
Run tests and generate html coverage report:
```console
python -m pytest --cov --cov-report html
```
