# -*- coding: utf-8 -*-
import socket
import json
import sys
import time
from datetime import datetime
import pytz

########## Change only this

# Group secret
secret = "g03o38k"
# Group port, change the last two digits to your group number
port = 39003
# Path to your python file containing the prediction
# sys.path.append("/ai/predict")

def get_date_and_time():
    from datetime import datetime
    date = datetime.now().strftime('%Y-%m-%d')
    hour = datetime.now().strftime('T%H')
    return date, hour

# Module(s) to perform prediction
from src.final_predictor import FinalPredictor

# Functions that are used to get your predictions
def predict_hour():
    date, hour = get_date_and_time()
    fp = FinalPredictor(date, hour)
    return float(fp.predict_hour())

def predict_day():
    date, hour = get_date_and_time()
    fp = FinalPredictor(date, hour)
    return float(fp.predict_day())

def predict_week():
    date, hour = get_date_and_time()
    fp = FinalPredictor(date, hour)
    return float(fp.predict_week())

##########

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    host = socket.gethostbyname(socket.gethostname() + '.local')
except socket.gaierror:
    host = socket.gethostbyname(socket.gethostname())
s.bind((host, port))
s.listen(5)

while True:
    conn, addr = s.accept()	# accept the connection
    conn.close()
    ts = str(datetime.now(pytz.timezone("Europe/Berlin")))[:-13]
    if addr[0] == "129.187.240.34":
        message = {"secret": secret, "time": ts, "hour": round(predict_hour(),4), "day": round(predict_day(),4), "week": round(predict_week(),4)}
        data = json.dumps(message)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print("Connecting to server.")
            sock.connect(("129.187.240.34", 39000))
            print("Sending data.")
            sock.sendall(bytes(data,encoding="utf-8"))
            print("Sent: {}".format(data))
            print("Data successfully sent.")
        except:
            print("Failed, retrying.")
            time.sleep(1)
        finally:
            print("Closing connection.")
            sock.close()
