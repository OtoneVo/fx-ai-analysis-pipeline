"""テクニカル指標の関数群（pandas のみ使用）"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """単純移動平均（SMA）"""
    result = series.rolling(window=window).mean()
    assert isinstance(result, pd.Series)
    return result


def ema(series: pd.Series, window: int) -> pd.Series:
    """指数移動平均（EMA）"""
    result = series.ewm(span=window, adjust=False).mean()
    assert isinstance(result, pd.Series)
    return result


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI（Wilder方式）"""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))
    assert isinstance(result, pd.Series)
    return result


def returns(series: pd.Series) -> pd.Series:
    """対数リターン: ln(P_t / P_{t-1})"""
    result = np.log(series / series.shift(1))
    assert isinstance(result, pd.Series)
    return result
