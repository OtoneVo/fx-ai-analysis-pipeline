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


def returns(series: pd.Series, period: int = 1) -> pd.Series:
    """対数リターン: ln(P_t / P_{t-period})"""
    result = np.log(series / series.shift(period))
    assert isinstance(result, pd.Series)
    return result


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD（ライン、シグナル、ヒストグラム）"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_band_width(series: pd.Series, window: int = 20) -> pd.Series:
    """ボリンジャーバンド幅（上バンド - 下バンド）/ 中央バンド"""
    mid = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    width = (2 * std) / mid
    assert isinstance(width, pd.Series)
    return width
