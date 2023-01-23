from venv import create
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
url = 'mongodb://{user}:{pw}@{host}:{port}/?replicaSet={rs}&authSource={auth_src}'.format(
    user=os.environ['mango_username'],
    pw=os.environ['mango_password'],
    host=os.environ['mango_host'],
    port=os.environ['mango_port'],
    rs='rs01',
    auth_src=os.environ['mango_database'])
ex = None


class save():
    pass
    # def __init__(self, data, file_name, exp_name, metrics=None) -> None:
    #     global ex
    #     ex = Experiment(exp_name)
    #     ex.observers.append(
    #         MongoObserver(url,
    #                       tlsCAFile=str(self.key_dir / 'cert.crt'),
    #                       db_name=os.environ['mango_database']))
    #     self.file = file_name
    #     self.data = data
    #     if metrics is None:
    #         metrics = self.data.columns.to_list()
    #     self.metrics = metrics
    #     self.push_to_mango()

    # @ex.main
    # def push_to_mango(self, _run):
    #     # data = pd.read_csv(str(data_dir / dname))
    #     for i, d in self.data[self.metrics].iterrows():
    #         for n, v in d.items():
    #             _run.log_scalar(n, v)


# def save_data(data):
#     data_dir.mkdir(exist_ok=True)
#     data.to_csv(str(data_dir / dname), index=False)

# def push(data, name, exp_name, fast_metrics):
#     global metrics, dname
#     metrics = fast_metrics
#     create_exp(exp_name)
#     save_data(data)
#     ex.run()
#     ex.add_artifact(str(data_dir / name))