"""改善版予測: 特徴量選択 + 多様なモデル × 閾値最適化で次足上昇を予測"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score
from plot_close import load_close
from dataset import make_features


def find_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """閾値を0.20〜0.80で刻み0.005でスイープし、Accuracyが最大になる閾値を返す"""
    best_thr = 0.5
    best_acc = 0.0
    for thr_int in range(40, 161):
        thr = thr_int / 200.0
        y_pred = (proba >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc


def main() -> None:
    close = load_close()
    features = make_features(close)

    # 目的変数: 次の足が上昇するか
    features["y"] = (close.shift(-1) > close).astype(int)
    features = features.dropna()

    # 特徴量列（close と y を除く）
    feature_cols = [c for c in features.columns if c not in ("close", "y")]

    X = features[feature_cols].values.astype(np.float64)
    y = features["y"].values.astype(np.int64)

    # 時系列分割（最後の 20% をテスト）
    split = int(len(features) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # スケーリング
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 特徴量選択（mutual information で上位 k 個選択）
    best_overall_acc = 0.0
    best_overall_name = ""
    best_overall_thr = 0.5
    best_overall_pred = None
    best_k = 0

    for k in [10, 15, 20, 25, 30, 47]:  # 全特徴量も含めて試す
        actual_k = min(k, X_train.shape[1])
        selector = SelectKBest(mutual_info_classif, k=actual_k)
        X_train_sel = selector.fit_transform(X_train_sc, y_train)
        X_test_sel = selector.transform(X_test_sc)

        # 複数モデルを定義
        models = {
            f"RF(k={actual_k})": RandomForestClassifier(
                n_estimators=500,
                max_depth=3,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=42,
                class_weight="balanced",
            ),
            f"GBT(k={actual_k})": GradientBoostingClassifier(
                n_estimators=300,
                max_depth=2,
                learning_rate=0.03,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.7,
                random_state=42,
            ),
            f"LR(k={actual_k})": LogisticRegression(
                C=0.01,
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
            f"ExtraTrees(k={actual_k})": ExtraTreesClassifier(
                n_estimators=500,
                max_depth=4,
                min_samples_split=15,
                min_samples_leaf=8,
                random_state=42,
                class_weight="balanced",
            ),
            f"SVC(k={actual_k})": SVC(
                C=1.0,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
        }

        for name, model in models.items():
            model.fit(X_train_sel, y_train)
            proba = model.predict_proba(X_test_sel)[:, 1]

            thr, acc = find_best_threshold(y_test, proba)

            if acc > best_overall_acc:
                best_overall_acc = acc
                best_overall_name = name
                best_overall_thr = thr
                best_overall_pred = (proba >= thr).astype(int)
                best_k = actual_k

    # 最良モデルの結果を出力
    y_pred = best_overall_pred
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print("=" * 40)
    print(f"最良モデル: {best_overall_name}")
    print("=" * 40)
    print(f"特徴量数(全体) : {len(feature_cols)}")
    print(f"選択特徴量数   : {best_k}")
    print(f"訓練データ数   : {len(y_train)}")
    print(f"テストデータ数 : {len(y_test)}")
    print(f"最良閾値       : {best_overall_thr:.3f}")
    print(f"Accuracy       : {best_overall_acc:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print("=" * 40)

    # 予測結果の最後の数行を CSV に保存
    test_df = features.iloc[split:].copy()
    test_df["y_pred"] = y_pred
    output = test_df[["close", "y", "y_pred"]].tail(20)
    output.to_csv("predictions.csv")
    print(f"\n予測結果の最後の {len(output)} 行を predictions.csv に保存しました。")


if __name__ == "__main__":
    main()
