"""ベースライン予測: 直前上昇なら次も上昇と予測し、accuracy/precision/recall を評価"""

import pandas as pd
from plot_close import load_close
from dataset import make_features


def accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """正解率"""
    return float((y_true == y_pred).sum()) / len(y_true)


def precision(y_true: pd.Series, y_pred: pd.Series) -> float:
    """適合率（予測が True のうち実際に True の割合）"""
    tp = int(((y_pred == True) & (y_true == True)).sum())  # noqa: E712
    fp = int(((y_pred == True) & (y_true == False)).sum())  # noqa: E712
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true: pd.Series, y_pred: pd.Series) -> float:
    """再現率（実際に True のうち予測も True の割合）"""
    tp = int(((y_pred == True) & (y_true == True)).sum())  # noqa: E712
    fn = int(((y_pred == False) & (y_true == True)).sum())  # noqa: E712
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def main() -> None:
    close = load_close()
    features = make_features(close)

    # 目的変数: 次の足が上昇するか
    features["y"] = close.shift(-1) > close
    features = features.dropna()

    # 予測: 直前リターンが正なら上昇と予測
    features["y_pred"] = features["return_1"] > 0

    # 時系列分割（最後の 20% をテスト）
    split = int(len(features) * 0.8)
    test = features.iloc[split:].copy()

    y_true = test["y"]
    y_pred = test["y_pred"]

    # 評価
    print("=" * 40)
    print("ベースライン予測 評価結果")
    print("=" * 40)
    print(f"テストデータ数 : {len(test)}")
    print(f"Accuracy       : {accuracy(y_true, y_pred):.4f}")
    print(f"Precision      : {precision(y_true, y_pred):.4f}")
    print(f"Recall         : {recall(y_true, y_pred):.4f}")
    print("=" * 40)

    # 予測結果の最後の数行を CSV に保存
    output = test[["close", "y", "y_pred"]].tail(20)
    output.to_csv("predictions.csv")
    print(f"\n予測結果の最後の {len(output)} 行を predictions.csv に保存しました。")


if __name__ == "__main__":
    main()
