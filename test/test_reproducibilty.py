import pandas as pd

df1 = pd.read_csv("top10_center_c15_pp_1.csv")
try:
    df2 = pd.read_csv("../top10_center_c15_pp_1.csv")
except FileNotFoundError:
    raise FileNotFoundError("Script needs to be run at least once. Refer to README.")

if df1["r_smiles"].to_list() == df2["r_smiles"].to_list():
    print("Same results.")
else:
    print("NOT same results.")