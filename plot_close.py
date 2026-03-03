import traceback

import pandas as pd
import matplotlib.pyplot as plt
import os


def _ensure_series(data: pd.DataFrame | pd.Series) -> pd.Series:
    """DataFrame または Series を受け取り、必ず Series を返す"""
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    assert isinstance(data, pd.Series)
    return data


def extract_close_column(df: pd.DataFrame) -> pd.Series:
    """DataFrameから 'Close' カラムを抽出する（マルチインデックス対応）"""
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            return _ensure_series(df.xs('Close', axis=1, level=0))
        cols = [c for c in df.columns if 'Close' in str(c)]
        if not cols:
            raise KeyError("カラム 'Close' が見つかりません。")
        return _ensure_series(df[cols[0]])

    if 'Close' not in df.columns:
        raise KeyError("カラム 'Close' が見つかりません。")
    return _ensure_series(df['Close'])


def load_close(csv_file: str = "usdjpy_1h_30d.csv") -> pd.Series:
    """CSV を読み込み Close 列を pd.Series として返す共通ヘルパー"""
    df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return extract_close_column(df)


def plot_usd_jpy_close():
    """USD/JPY の終値を時系列グラフで表示する"""
    csv_file = "usdjpy_1h_30d.csv"

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} が見つかりません。")
        return

    try:
        df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0, parse_dates=True)

        # インデックスを DatetimeIndex に変換（必要な場合のみ）
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        close = extract_close_column(df)

        # グラフのプロット
        plt.figure(figsize=(12, 6))
        plt.plot(close.index.to_numpy(), close.to_numpy(), label='USD/JPY Close Price')
        plt.title('USD/JPY 1h Close Price - Last 30 Days', fontsize=14)
        plt.xlabel('Datetime', fontsize=12)
        plt.ylabel('Price (JPY)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        print("グラフを表示します。ウィンドウを閉じるとプログラムを終了します。")
        plt.show()

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    plot_usd_jpy_close()
