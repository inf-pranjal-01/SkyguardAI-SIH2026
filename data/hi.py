import pandas as pd
df = pd.read_csv("AWS-MUM-101.csv", parse_dates=["timestamp"])
diffs = df["pressure_hpa"].diff().abs()
top5_idx = diffs.nlargest(5).index
print(df.loc[top5_idx.union(top5_idx-1).union(top5_idx+1)].sort_values("timestamp")[["timestamp","pressure_hpa"]])