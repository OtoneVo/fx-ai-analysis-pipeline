"""特徴量 DataFrame を生成し features.csv として保存する"""

import pandas as pd
from plot_close import load_close
from indicators import sma, ema, rsi, returns, macd, bollinger_band_width


def make_features(close: pd.Series) -> pd.DataFrame:
    """Close から学習用の特徴量 DataFrame を生成する"""
    df = pd.DataFrame({"close": close})

    # 基本リターン
    df["return_1"] = returns(close)
    df["return_2"] = returns(close, period=2)
    df["return_4"] = returns(close, period=4)
    df["return_8"] = returns(close, period=8)

    # 移動平均
    df["sma20"] = sma(close, 20)
    df["sma50"] = sma(close, 50)
    df["ema20"] = ema(close, 20)

    # 移動平均比率（価格との乖離）
    df["close_sma20_ratio"] = close / df["sma20"]
    df["close_sma50_ratio"] = close / df["sma50"]
    df["sma20_sma50_ratio"] = df["sma20"] / df["sma50"]

    # RSI
    df["rsi14"] = rsi(close, 14)
    df["rsi14_diff"] = df["rsi14"].diff()

    # ボラティリティ
    df["volatility"] = close.rolling(window=20).std()
    df["volatility_ratio"] = df["volatility"] / close

    # MACD
    macd_line, signal_line, histogram = macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram

    # ボリンジャーバンド幅
    df["bb_width"] = bollinger_band_width(close, 20)

    # ラグ特徴量（直近のパターンを記憶させる）
    for lag in [1, 2, 3, 4]:
        df[f"return_1_lag{lag}"] = df["return_1"].shift(lag)
        df[f"rsi14_lag{lag}"] = df["rsi14"].shift(lag)
        df[f"macd_hist_lag{lag}"] = df["macd_hist"].shift(lag)

    # 直近N本の上昇割合
    df["up_ratio_5"] = (df["return_1"] > 0).rolling(5).mean()
    df["up_ratio_10"] = (df["return_1"] > 0).rolling(10).mean()
    df["up_ratio_20"] = (df["return_1"] > 0).rolling(20).mean()

    # リターンの統計量
    df["return_std_5"] = df["return_1"].rolling(5).std()
    df["return_std_10"] = df["return_1"].rolling(10).std()
    df["return_mean_5"] = df["return_1"].rolling(5).mean()
    df["return_mean_10"] = df["return_1"].rolling(10).mean()

    # 短期EMAと比率
    df["ema5"] = ema(close, 5)
    df["ema10"] = ema(close, 10)
    df["close_ema5_ratio"] = close / df["ema5"]
    df["close_ema10_ratio"] = close / df["ema10"]
    df["ema5_ema20_ratio"] = df["ema5"] / df["ema20"]

    # 価格変化の二次差分（加速度）
    df["return_accel"] = df["return_1"].diff()

    # RSI zones（過買い・過売りの数値化）
    df["rsi_overbought"] = (df["rsi14"] > 70).astype(int)
    df["rsi_oversold"] = (df["rsi14"] < 30).astype(int)

    # 時間帯・曜日特徴量（日時インデックスがある場合）
    if hasattr(close.index, "hour"):
        df["hour"] = close.index.hour
        df["dayofweek"] = close.index.dayofweek

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
