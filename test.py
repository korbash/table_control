from mylib.sacred_logger import ex, save_data
import pandas as pd
import pymongo
import dotenv
import warnings
import os
from pathlib import Path

# warnings.filterwarnings("ignore", category=FutureWarning)

# key_dir = Path('keys')
# dotenv.load_dotenv(key_dir / 'keys.env')

# print(os.environ['host'])
# client = pymongo.MongoClient(host=os.environ['host'],
#                              port=27018,
#                              replicaSet='rs01',
#                              username=os.environ['username'],
#                              password=os.environ['password'],
#                              authSource=os.environ['database'],
#                              tls=True,
#                              tlsCAFile=str(key_dir / 'cert.crt'))
# db = client[os.environ['database']]
# print(db.list_collection_names())
df = pd.DataFrame({'x': [2,3], 'y':[4,1]})
print(df)
save_data(df)
ex.run()
ex.add_artifact('data/pull_resalts.csv')
