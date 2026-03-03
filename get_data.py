import sys
import yfinance as yf
import pandas as pd

print("PY:", sys.executable)

data = yf.download("JPY=X", period="30d", interval="1h")

# ここを追加：型をDataFrameに寄せる & None対策
if data is None or data.empty:
    raise RuntimeError("データ取得に失敗しました（空データ）")

df: pd.DataFrame = data
print(df.head())

df.to_csv("usdjpy_1h_30d.csv", index=True)
print("saved: usdjpy_1h_30d.csv")