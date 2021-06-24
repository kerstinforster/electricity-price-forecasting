<div style="text-align: right; margin-top: 10px; margin-bottom: -30px;"> 
<img height="50px" class="center-block" src="https://www.tum.de/typo3conf/ext/in2template/Resources/Public/Images/tum-logo.svg">
</div>

# Group 3 Applied Machine Intelligence

This is the project repository of group 3 for the course Applied Machine Intelligence at TUM.  
Group Members:
- Adrian Michl (MTK: )
- Anna Mrosik (MTK: 03695133)
- Farouk Ferjani (MTK: )
- Florian Mysliwetz (MTK: 03671374)
- Jakob Kruse (MTK: 03682020)
- Kerstin Forster (MTK: )
- Oliver Hehmann (MTK: )
- Philip Durst (MTK: )

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

## Execution
Run the code:
TBD


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
