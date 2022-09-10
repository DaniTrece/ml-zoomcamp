import numpy as np
import pandas as pd

print(f"1.- np.__version__: {np.__version__}")

df = pd.read_csv("./data/car-price.csv")
print(f"2.- number of rows: {len(df)}")


make_count_series = df.groupby("Make").Model.count()
sol_3 = make_count_series.sort_values(ascending=False).head(n=3)
print(f"3.- top-3: \n {sol_3}")


audis = df[df["Make"] == "Audi"]
sol_4 = len(audis["Model"].drop_duplicates())
print(f"4.- unique Audi car models: {sol_4}")


nulls_by_column = df.isnull().sum()
sol_5 = len(nulls_by_column[nulls_by_column > 0])
print(f"5.- # columns have missing vals: {sol_5}")


ec = df["Engine Cylinders"]
ec_median = ec.median()  # 6.0
ec_mode = ec.mode()[0]  # 4.0
df_filled = df.fillna(value={"Engine Cylinders": 4.0})
ec_filled = df_filled["Engine Cylinders"]
ec_filled.median()  # 6.0
print(f"6.- Has it changed?: {ec_filled.median()!=ec_median}")

lotus = df[df["Make"] == "Lotus"]
s = lotus[["Engine HP", "Engine Cylinders"]]
s_no_dup = s.drop_duplicates()
X = np.array(s_no_dup)
XTX = np.dot(X.T, X)
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])
w = np.dot(np.dot(XTX_inv, X.T), y)  # [4.59494481, -63.56432501]
print(f"7.- w[0] = {w[0]}")
