import numpy as np
import pandas as pd
import os
import dotenv
from pathlib import Path
from sacred.observers import MongoObserver
from sacred import Experiment

main_dir = Path(__file__).parents[1]
key_dir = main_dir / 'keys'
data_dir = main_dir / 'data'
dotenv.load_dotenv(key_dir / 'keys.env')

# print(os.environ['host'])

url = 'mongodb://{user}:{pw}@{host}:{port}/?replicaSet={rs}&authSource={auth_src}'.format(
    user=os.environ['username'],
    pw=os.environ['password'],
    host=os.environ['host'],
    port=os.environ['port'],
    rs='rs01',
    auth_src='db1')

ex = Experiment('test')
ex.observers.append(
    MongoObserver(url,
                  tlsCAFile=str(key_dir / 'cert.crt'),
                  db_name=os.environ['database']))

# @ex.capture
# def live_metrics(params, _run):
#     for name, val in params.items():
#         _run.log_scalar(name, val)


@ex.automain
def push_to_mango():
    pass


def save_data(data, name):
    data_dir.mkdir(exist_ok=True)
    data.to_csv(str(data_dir / name), index=False)
