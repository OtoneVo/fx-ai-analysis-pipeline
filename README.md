# FX Analysis Pipeline

## 目的
USD/JPY 1時間足データを用いて、
特徴量生成・可視化・ベースライン予測までを構築。

## 使用技術
- Python
- pandas
- matplotlib
- 時系列分割

## 構成
- indicators.py
- dataset.py
- plot_chart.py
- baseline_predict.py

## 結果
Accuracy: 0.4797（ベースライン）
→ 単純ルールでは予測困難であることを確認

## 今後の改善案
- 特徴量追加
- モデル高度化（LightGBM等）
- EA連携