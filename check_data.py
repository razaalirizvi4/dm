import pandas as pd
from pathlib import Path
DATA_PATH = Path(r'D:/work/dmProj/AI_Powered_IoT_Network_Intrusion_Detection_Dataset.csv')
df = pd.read_csv(DATA_PATH, nrows=100)
print("Columns and Dtypes:")
print(df.dtypes)
print("\nTarget Column Value Counts:")
target = 'intrusion_label' # assuming this from the notebook
if target in df.columns:
    print(df[target].value_counts())
else:
    print(f"Column '{target}' not found. Available columns: {df.columns.tolist()}")
