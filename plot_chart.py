"""Close + SMA チャートと RSI チャートを別ウィンドウで表示する"""

import matplotlib.pyplot as plt
from plot_close import load_close
from indicators import sma, rsi


def main() -> None:
    close = load_close()

    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    rsi14 = rsi(close, 14)

    # --- 図1: Close + SMA ---
    plt.figure(figsize=(12, 6))
    plt.plot(close.index.to_numpy(), close.to_numpy(), label="Close", linewidth=1)
    plt.plot(sma20.index.to_numpy(), sma20.to_numpy(), label="SMA(20)", linewidth=1)
    plt.plot(sma50.index.to_numpy(), sma50.to_numpy(), label="SMA(50)", linewidth=1)
    plt.title("USD/JPY 1H - Close & SMA", fontsize=14)
    plt.xlabel("Datetime", fontsize=12)
    plt.ylabel("Price (JPY)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # --- 図2: RSI（別ウィンドウ） ---
    plt.figure(figsize=(12, 4))
    plt.plot(rsi14.index.to_numpy(), rsi14.to_numpy(), label="RSI(14)", color="purple", linewidth=1)
    plt.axhline(70, color="red", linestyle="--", alpha=0.5, label="Overbought (70)")
    plt.axhline(30, color="green", linestyle="--", alpha=0.5, label="Oversold (30)")
    plt.title("USD/JPY 1H - RSI(14)", fontsize=14)
    plt.xlabel("Datetime", fontsize=12)
    plt.ylabel("RSI", fontsize=12)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    print("グラフを表示します。ウィンドウを閉じるとプログラムを終了します。")
    plt.show()


if __name__ == "__main__":
    main()
