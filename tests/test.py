from tkinter import N
from mylib.sacred_logger import ex, save_data
import pandas as pd
import pymongo
import dotenv
import warnings
import os
from pathlib import Path

# warnings.filterwarnings("ignore", category=FutureWarning)

key_dir = Path('keys')
dotenv.load_dotenv(key_dir / 'keys.env')

# print(os.environ['mango_host'])
# print(os.environ['mango_username'])
# print(os.environ['mango_password'])
# print(os.environ['mango_database'])
# client = pymongo.MongoClient(host=os.environ['mango_host'],
#                              port=int(os.environ['mango_port']),
#                              replicaSet='rs01',
#                              username=os.environ['mango_username'],
#                              password=os.environ['mango_password'],
#                              authSource=os.environ['mango_database'],
#                              tls=True,
#                              tlsCAFile=str(key_dir / 'cert.crt'))
# db = client[os.environ['mango_database']]
# print(db.list_collection_names())
 

df = pd.DataFrame({'x': [2,3], 'y':[4,1]})
print(df)
save_data(df, name='pull_resalts.csv')
ex.run()
ex.add_artifact('data/pull_resalts.csv')
