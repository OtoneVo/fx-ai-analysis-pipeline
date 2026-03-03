"""特徴量 DataFrame を生成し features.csv として保存する"""

import pandas as pd
from plot_close import load_close
from indicators import sma, ema, rsi, returns


def make_features(close: pd.Series) -> pd.DataFrame:
    """Close から学習用の特徴量 DataFrame を生成する"""
    df = pd.DataFrame({"close": close})
    df["return_1"] = returns(close)
    df["sma20"] = sma(close, 20)
    df["sma50"] = sma(close, 50)
    df["ema20"] = ema(close, 20)
    df["rsi14"] = rsi(close, 14)
    df["volatility"] = close.rolling(window=20).std()
    return df.dropna()


def save_features(df: pd.DataFrame, path: str = "features.csv") -> None:
    """特徴量 DataFrame を CSV に保存する"""
    df.to_csv(path)
    print(f"特徴量を {path} に保存しました（{len(df)} 行）")


def main() -> None:
    close = load_close()
    features = make_features(close)
    print(features.head(10))
    print(f"\n特徴量数: {features.shape[1]}, データ数: {features.shape[0]}")
    save_features(features)


if __name__ == "__main__":
    main()
