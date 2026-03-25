import pandas as pd
from pathlib import Path
DATA_PATH = Path(r'D:/work/dmProj/AI_Powered_IoT_Network_Intrusion_Detection_Dataset.csv')
df = pd.read_csv(DATA_PATH, nrows=5)
print(df.columns.tolist())
print(df.dtypes)
