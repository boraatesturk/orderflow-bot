"""
ORDERFLOW BOT - ML Model v2 (XGBoost)
======================================
Kullanim:
    python ml_model.py --train          # Her iki modeli de egit, karsilastir
    python ml_model.py --predict        # Son bar icin her iki modelden tahmin

Target: 3 bar (15 dk) sonra fiyat yonu
    UP   (+1) : %+0.15 uzeri artis
    FLAT ( 0) : -%0.15 ile +%0.15 arasi  (binary modda kullanilmaz)
    DOWN (-1) : -%0.15 alti dusus

Threshold degistirmek icin: TARGET_THRESHOLD = 0.0015
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
except ImportError:
    print("[HATA] xgboost kurulu degil. Calistir: pip install xgboost")
    raise

warnings.filterwarnings("ignore")

# ─── AYARLAR ────────────────────────────────────────────────────────────────
PARQUET_PATH = Path("data/ETHUSDT_orderflow_365d_5m.parquet")
MODEL_W_PATH = Path("data/xgb_weighted.pkl")
MODEL_B_PATH = Path("data/xgb_binary.pkl")

TARGET_BARS      = 3
TARGET_THRESHOLD = 0.003
TRAIN_RATIO      = 0.70

LABEL_MAP     = {-1: "DOWN", 0: "FLAT", 1: "UP"}
CLASS_WEIGHTS = {-1: 2.0, 0: 0.5, 1: 2.0}

XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "eval_metric":      "mlogloss",
    "random_state":     42,
    "n_jobs":           -1,
}
# ────────────────────────────────────────────────────────────────────────────


def load_data():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet bulunamadi: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    df.sort_index(inplace=True)
    print(f"[+] Veri yuklendi: {len(df):,} bar | {df.index[0]} -> {df.index[-1]}")
    return df


def build_features(df):
    feat = df.copy()
    lag_cols = ["delta", "imbalance_ratio", "cvd", "volume", "close"]
    for col in lag_cols:
        if col not in feat.columns:
            continue
        for lag in [1, 3, 6, 12]:
            feat[f"{col}_lag{lag}"] = feat[col].shift(lag)

    for window in [5, 10, 20]:
        feat[f"close_roc_{window}"]    = feat["close"].pct_change(window)
        feat[f"volume_mean_{window}"]  = feat["volume"].rolling(window).mean()
        feat[f"delta_mean_{window}"]   = feat["delta"].rolling(window).mean()
        feat[f"delta_std_{window}"]    = feat["delta"].rolling(window).std()
        feat[f"imbalance_ma_{window}"] = feat["imbalance_ratio"].rolling(window).mean()

    if "vwap" in feat.columns:
        feat["price_vs_vwap"] = (feat["close"] - feat["vwap"]) / feat["vwap"]
    if "volume" in feat.columns and "delta" in feat.columns:
        feat["delta_norm"] = feat["delta"] / (feat["volume"] + 1e-9)
    if "cvd" in feat.columns:
        feat["cvd_roc5"]  = feat["cvd"].diff(5)
        feat["cvd_roc10"] = feat["cvd"].diff(10)

    feat["bar_direction"] = np.sign(feat["close"] - feat["open"])
    feat["hl_range"]      = (feat["high"] - feat["low"]) / feat["close"]

    bool_cols = feat.select_dtypes(include="bool").columns
    feat[bool_cols] = feat[bool_cols].astype(int)
    return feat


def build_target(df):
    future_close = df["close"].shift(-TARGET_BARS)
    pct_change   = (future_close - df["close"]) / df["close"]
    target = pd.Series(0, index=df.index, name="target")
    target[pct_change >  TARGET_THRESHOLD] =  1
    target[pct_change < -TARGET_THRESHOLD] = -1
    return target


def prepare_dataset(df):
    feat   = build_features(df)
    target = build_target(df)

    combined = feat.copy()
    combined["__target__"] = target
    combined.dropna(inplace=True)

    non_feat = ["open", "high", "low", "close",
                "typical_price", "poc_price", "__target__"]
    feature_cols = [
        c for c in combined.columns
        if c not in non_feat and pd.api.types.is_numeric_dtype(combined[c])
    ]

    X = combined[feature_cols]
    y = combined["__target__"]

    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[+] Feature sayisi : {len(feature_cols)}")
    print(f"[+] Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"[+] Target dagilimi:")
    for lbl, name in LABEL_MAP.items():
        n = (y == lbl).sum()
        print(f"    {name:>4}: {n:,} ({n/len(y)*100:.1f}%)")

    return X_train, X_test, y_train, y_test, feature_cols


# ─── MODEL 1: WEIGHTED ───────────────────────────────────────────────────────
def train_weighted(X_train, y_train):
    le    = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    sw    = np.array([CLASS_WEIGHTS[v] for v in y_train])

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_enc, sample_weight=sw,
              eval_set=[(X_train, y_enc)], verbose=False)
    print("[+] Weighted model egitimi tamamlandi.")
    return model, le


def evaluate_weighted(model, le, X_test, y_test):
    y_enc  = le.transform(y_test)
    y_pred = model.predict(X_test)
    labels = [LABEL_MAP[c] for c in le.classes_]

    print(f"\n{'='*52}")
    print(f"  WEIGHTED - SINIFLANDIRMA RAPORU")
    print(f"{'='*52}")
    print(classification_report(y_enc, y_pred, target_names=labels))
    acc = (y_pred == y_enc).mean()
    print(f"[+] Test Accuracy: {acc*100:.2f}%")

    rep = classification_report(y_enc, y_pred, target_names=labels, output_dict=True)
    _save_confusion(y_enc, y_pred, labels, "data/confusion_weighted.png", "WEIGHTED")
    return acc, rep


# ─── MODEL 2: BINARY ─────────────────────────────────────────────────────────
def train_binary(X_train, y_train):
    mask  = y_train != 0
    X_tr  = X_train[mask]
    y_tr  = y_train[mask]

    le    = LabelEncoder()
    y_enc = le.fit_transform(y_tr)

    params = {**XGB_PARAMS, "eval_metric": "logloss"}
    model  = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_enc, eval_set=[(X_tr, y_enc)], verbose=False)
    print("[+] Binary model egitimi tamamlandi.")
    return model, le


def evaluate_binary(model, le, X_test, y_test):
    mask   = y_test != 0
    X_te   = X_test[mask]
    y_te   = y_test[mask]
    print(f"\n[i] Binary test: {len(X_te):,} bar (FLAT {(y_test==0).sum():,} cikarildi)")

    y_enc  = le.transform(y_te)
    y_pred = model.predict(X_te)
    labels = [LABEL_MAP[c] for c in le.classes_]

    print(f"\n{'='*52}")
    print(f"  BINARY - SINIFLANDIRMA RAPORU")
    print(f"{'='*52}")
    print(classification_report(y_enc, y_pred, target_names=labels))
    acc = (y_pred == y_enc).mean()
    print(f"[+] Test Accuracy: {acc*100:.2f}%")

    rep = classification_report(y_enc, y_pred, target_names=labels, output_dict=True)
    _save_confusion(y_enc, y_pred, labels, "data/confusion_binary.png", "BINARY")
    return acc, rep


# ─── YARDIMCI ────────────────────────────────────────────────────────────────
def _save_confusion(y_true, y_pred, labels, path, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {title}", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[+] Confusion matrix: {path}")


def plot_feature_importance(model_w, model_b, feature_cols, top_n=25):
    fi_w = pd.Series(model_w.feature_importances_, index=feature_cols)
    fi_b = pd.Series(model_b.feature_importances_, index=feature_cols)

    top_w = fi_w.sort_values(ascending=False).head(top_n)
    top_b = fi_b.sort_values(ascending=False).head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(18, top_n * 0.38 + 1.5))
    for ax, top, title in zip(axes, [top_w, top_b],
                               ["Weighted (3-sinif)", "Binary (UP/DOWN)"]):
        colors = ["#2ecc71" if i < top_n*0.33
                  else "#f39c12" if i < top_n*0.66
                  else "#e74c3c" for i in range(len(top))]
        ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor="none")
        ax.axvline(top.values.mean(), color="gray", linestyle="--", alpha=0.6)
        ax.set_title(f"Feature Importance - {title}", fontsize=12, pad=10)
        ax.set_xlabel("Importance (gain)")

    plt.suptitle("Model Karsilastirma - Feature Importance", fontsize=14, y=1.01)
    plt.tight_layout()
    path = Path("data/feature_importance_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Feature importance grafigi: {path}")

    print(f"\n{'TOP 15':^55}")
    print(f"{'Rank':<5} {'Weighted Feature':<30} {'Binary Feature':<30}")
    print("-"*65)
    w_top15 = fi_w.sort_values(ascending=False).head(15).index
    b_top15 = fi_b.sort_values(ascending=False).head(15).index
    for i, (wf, bf) in enumerate(zip(w_top15, b_top15), 1):
        print(f"{i:<5} {wf:<30} {bf:<30}")


def print_comparison_summary(acc_w, rep_w, acc_b, rep_b):
    print(f"\n{'='*55}")
    print(f"  KARSILASTIRMA OZETI")
    print(f"{'='*55}")
    print(f"  {'Metrik':<28} {'Weighted':>10} {'Binary':>10}")
    print(f"  {'-'*48}")
    print(f"  {'Accuracy':<28} {acc_w*100:>9.1f}% {acc_b*100:>9.1f}%")
    for lbl in ["DOWN", "UP"]:
        if lbl in rep_w and lbl in rep_b:
            print(f"  {lbl+' precision':<28} "
                  f"{rep_w[lbl]['precision']*100:>9.1f}% "
                  f"{rep_b[lbl]['precision']*100:>9.1f}%")
            print(f"  {lbl+' recall':<28} "
                  f"{rep_w[lbl]['recall']*100:>9.1f}% "
                  f"{rep_b[lbl]['recall']*100:>9.1f}%")
    print(f"{'='*55}")
    winner = "Weighted" if acc_w >= acc_b else "Binary"
    print(f"\n  Accuracy kazanani  : {winner}")
    print(f"  Tavsiye            : Binary daha pratik (FLAT filtreli)")


def save_model(model, le, feature_cols, path):
    payload = {"model": model, "le": le, "feature_cols": feature_cols}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[+] Model kaydedildi: {path}")


def load_model(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model bulunamadi: {path}  -> Once --train calistir.")
    with open(path, "rb") as f:
        p = pickle.load(f)
    return p["model"], p["le"], p["feature_cols"]


def predict_latest(df, model_w, le_w, model_b, le_b, feature_cols):
    feat = build_features(df)
    last = feat[feature_cols].iloc[[-1]].fillna(0)

    prob_w     = model_w.predict_proba(last)[0]
    pred_w_lbl = LABEL_MAP[le_w.inverse_transform([model_w.predict(last)[0]])[0]]

    prob_b     = model_b.predict_proba(last)[0]
    pred_b_lbl = LABEL_MAP[le_b.inverse_transform([model_b.predict(last)[0]])[0]]

    print("\n" + "="*48)
    print("  SON BAR TAHMINI")
    print("="*48)
    print(f"  Zaman  : {df.index[-1]}")
    print(f"  Fiyat  : {df['close'].iloc[-1]:.2f}")
    print(f"\n  [WEIGHTED]  {pred_w_lbl}")
    print(f"  DOWN={prob_w[0]:.1%}  FLAT={prob_w[1]:.1%}  UP={prob_w[2]:.1%}")
    print(f"\n  [BINARY]    {pred_b_lbl}")
    print(f"  DOWN={prob_b[0]:.1%}  UP={prob_b[1]:.1%}")
    print("="*48)


# ─── ANA AKIS ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="OrderFlow ML Model v2")
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    if not any([args.train, args.predict]):
        parser.print_help()
        return

    df = load_data()

    if args.train:
        X_train, X_test, y_train, y_test, feature_cols = prepare_dataset(df)

        model_w, le_w = train_weighted(X_train, y_train)
        acc_w, rep_w  = evaluate_weighted(model_w, le_w, X_test, y_test)
        save_model(model_w, le_w, feature_cols, MODEL_W_PATH)

        model_b, le_b = train_binary(X_train, y_train)
        acc_b, rep_b  = evaluate_binary(model_b, le_b, X_test, y_test)
        save_model(model_b, le_b, feature_cols, MODEL_B_PATH)

        plot_feature_importance(model_w, model_b, feature_cols)
        print_comparison_summary(acc_w, rep_w, acc_b, rep_b)

    if args.predict:
        model_w, le_w, feature_cols = load_model(MODEL_W_PATH)
        model_b, le_b, _            = load_model(MODEL_B_PATH)
        predict_latest(df, model_w, le_w, model_b, le_b, feature_cols)


if __name__ == "__main__":
    main()