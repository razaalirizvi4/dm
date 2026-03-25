import pandas as pd

df = pd.read_csv(r'D:\work\dmProj\AI_Powered_IoT_Network_Intrusion_Detection_Dataset.csv')
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nDtypes:\n{df.dtypes.to_string()}")

# Check unique values for categorical columns
print("\n=== Column Info ===")
for col in df.columns:
    n_unique = df[col].nunique()
    dtype = df[col].dtype
    if n_unique <= 20:
        vals = df[col].unique().tolist()
        print(f"{col}: {n_unique} unique, dtype={dtype}, values={vals}")
    else:
        print(f"{col}: {n_unique} unique, dtype={dtype}")

print(f"\nMissing: {df.isnull().sum().sum()}")

# Write full info to file
with open('_data_info.txt', 'w') as f:
    f.write(f"Shape: {df.shape}\n")
    f.write(f"Columns: {df.columns.tolist()}\n\n")
    f.write(f"Dtypes:\n{df.dtypes.to_string()}\n\n")
    for col in df.columns:
        n_unique = df[col].nunique()
        dtype = df[col].dtype
        if n_unique <= 20:
            vals = df[col].unique().tolist()
            f.write(f"{col}: {n_unique} unique, dtype={dtype}, values={vals}\n")
        else:
            f.write(f"{col}: {n_unique} unique, dtype={dtype}\n")
    f.write(f"\nMissing values total: {df.isnull().sum().sum()}\n")
    f.write(f"\nMissing per column:\n{df.isnull().sum().to_string()}\n")
    f.write(f"\nHead:\n{df.head(5).to_string()}\n")
    f.write(f"\nDescribe:\n{df.describe().to_string()}\n")

print("Written to _data_info.txt")
