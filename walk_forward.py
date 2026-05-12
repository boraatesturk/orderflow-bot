"""
ORDERFLOW BOT - Walk-Forward Backtest
======================================
Kullanim:
    python walk_forward.py

Nasil calisir:
    - Veriyi kronolojik pencerelerle boler
    - Her pencerede: TRAIN -> TEST -> precision/accuracy kaydet
    - Kac pencerede model rasgele tahmin seviyesinin ustunde?
    - Overfitting var mi? Her donem tutarli mi?

Parametreler:
    TRAIN_MONTHS : Her fold'un train suresi (ay)
    TEST_MONTHS  : Her fold'un test suresi (ay)
    STEP_MONTHS  : Pencere kayma adimi (ay)
"""

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
except ImportError:
    print("[HATA] xgboost kurulu degil: pip install xgboost")
    raise

warnings.filterwarnings("ignore")

# ─── AYARLAR ────────────────────────────────────────────────────────────────
PARQUET_PATH     = Path("data/ETHUSDT_orderflow_365d_5m.parquet")
TARGET_BARS      = 3
TARGET_THRESHOLD = 0.003

TRAIN_MONTHS = 3      # Her fold: 3 ay egit
TEST_MONTHS  = 1      # Her fold: 1 ay test et
STEP_MONTHS  = 1      # Her adimda 1 ay ileri kay

XGB_PARAMS = {
    "n_estimators":     300,   # Walk-forward icin biraz daha hizli
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}
# ────────────────────────────────────────────────────────────────────────────


def load_data():
    df = pd.read_parquet(PARQUET_PATH)
    df.sort_index(inplace=True)
    print(f"[+] Veri: {len(df):,} bar | {df.index[0].date()} -> {df.index[-1].date()}")
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


def get_feature_cols(df):
    non_feat = ["open", "high", "low", "close",
                "typical_price", "poc_price", "__target__"]
    return [
        c for c in df.columns
        if c not in non_feat and pd.api.types.is_numeric_dtype(df[c])
    ]


def make_folds(df):
    """Kronolojik pencereler uret."""
    start = df.index[0]
    end   = df.index[-1]

    folds = []
    train_start = start

    while True:
        train_end  = train_start + pd.DateOffset(months=TRAIN_MONTHS)
        test_start = train_end
        test_end   = test_start + pd.DateOffset(months=TEST_MONTHS)

        if test_end > end:
            break

        folds.append({
            "train_start": train_start,
            "train_end":   train_end,
            "test_start":  test_start,
            "test_end":    test_end,
        })
        train_start += pd.DateOffset(months=STEP_MONTHS)

    print(f"[+] {len(folds)} fold olusturuldu "
          f"(train={TRAIN_MONTHS}ay, test={TEST_MONTHS}ay, adim={STEP_MONTHS}ay)")
    return folds


def run_fold(feat_df, target, feature_cols, fold, idx):
    ts, te = fold["train_start"], fold["train_end"]
    vs, ve = fold["test_start"],  fold["test_end"]

    combined = feat_df.copy()
    combined["__target__"] = target
    combined.dropna(inplace=True)

    train = combined.loc[ts:te]
    test  = combined.loc[vs:ve]

    if len(train) < 200 or len(test) < 50:
        return None   # Yeterli veri yok

    X_train = train[feature_cols]
    y_train = train["__target__"]
    X_test  = test[feature_cols]
    y_test  = test["__target__"]

    # Binary: FLAT cikar
    tr_mask = y_train != 0
    te_mask = y_test  != 0
    X_train, y_train = X_train[tr_mask], y_train[tr_mask]
    X_test,  y_test  = X_test[te_mask],  y_test[te_mask]

    if len(X_test) < 30:
        return None

    le    = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_enc, eval_set=[(X_train, y_enc)], verbose=False)

    y_te_enc  = le.transform(y_test)
    y_pred    = model.predict(X_test)
    probs     = model.predict_proba(X_test)

    acc = (y_pred == y_te_enc).mean()

    classes   = le.classes_
    down_idx  = list(classes).index(-1)
    up_idx    = list(classes).index(1)

    rep = classification_report(y_te_enc, y_pred,
                                target_names=["DOWN", "UP"],
                                output_dict=True, zero_division=0)

    # DOWN bias orani
    up_signals = (probs[:, up_idx] >= 0.60).sum()
    dn_signals = (probs[:, down_idx] >= 0.60).sum()
    bias_ratio = dn_signals / max(up_signals, 1)

    result = {
        "fold"        : idx + 1,
        "train_start" : ts.date(),
        "train_end"   : te.date(),
        "test_start"  : vs.date(),
        "test_end"    : ve.date(),
        "train_bars"  : len(X_train),
        "test_bars"   : len(X_test),
        "accuracy"    : round(acc * 100, 1),
        "up_prec"     : round(rep.get("UP",   {}).get("precision", 0) * 100, 1),
        "dn_prec"     : round(rep.get("DOWN", {}).get("precision", 0) * 100, 1),
        "up_recall"   : round(rep.get("UP",   {}).get("recall",    0) * 100, 1),
        "dn_recall"   : round(rep.get("DOWN", {}).get("recall",    0) * 100, 1),
        "up_signals"  : int(up_signals),
        "dn_signals"  : int(dn_signals),
        "bias_ratio"  : round(bias_ratio, 2),
    }

    status = "✓" if acc > 0.52 else "✗"
    print(f"  Fold {idx+1:>2} [{vs.date()} - {ve.date()}] "
          f"Acc={acc*100:.1f}%  UP={result['up_prec']}%  DN={result['dn_prec']}%  "
          f"Bias={bias_ratio:.1f}x  {status}")
    return result


def print_summary(results):
    df = pd.DataFrame(results)

    print(f"\n{'='*75}")
    print(f"  WALK-FORWARD OZET")
    print(f"{'='*75}")
    print(f"  {'Fold':>5} {'Test donemi':<14} {'Accuracy':>10} "
          f"{'UP Prec':>9} {'DN Prec':>9} {'Bias':>7}")
    print(f"  {'-'*65}")
    for _, r in df.iterrows():
        marker = " ✗" if r["accuracy"] < 52 else ""
        print(f"  {int(r['fold']):>5} {str(r['test_start']):<14} "
              f"{r['accuracy']:>9.1f}% "
              f"{r['up_prec']:>8.1f}% "
              f"{r['dn_prec']:>8.1f}% "
              f"{r['bias_ratio']:>6.1f}x"
              f"{marker}")

    print(f"{'='*75}")
    print(f"\n  ORTALAMALAR:")
    print(f"  Accuracy   : {df['accuracy'].mean():.1f}%  "
          f"(std: {df['accuracy'].std():.1f}pp)")
    print(f"  UP Prec    : {df['up_prec'].mean():.1f}%  "
          f"(std: {df['up_prec'].std():.1f}pp)")
    print(f"  DOWN Prec  : {df['dn_prec'].mean():.1f}%  "
          f"(std: {df['dn_prec'].std():.1f}pp)")
    print(f"  DOWN bias  : {df['bias_ratio'].mean():.1f}x  "
          f"(min: {df['bias_ratio'].min():.1f}x  max: {df['bias_ratio'].max():.1f}x)")

    good_folds = (df["accuracy"] > 52).sum()
    total      = len(df)
    print(f"\n  %52 ustunde fold: {good_folds}/{total} "
          f"({good_folds/total*100:.0f}%)")

    if df["accuracy"].std() > 5:
        print("\n  [!] UYARI: Accuracy std > 5pp — model doneme cok duyarli (overfitting riski)")
    else:
        print("\n  [+] Accuracy std makul — model doneme gore cok degismiyor")

    if df["bias_ratio"].mean() > 1.5:
        print(f"  [!] UYARI: DOWN bias tutarli ({df['bias_ratio'].mean():.1f}x) "
              f"— class weight duzeltmesi gerekiyor")


def plot_results(results):
    df  = pd.DataFrame(results)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b9cad")
        ax.spines[:].set_color("#2d3748")

    x     = range(len(df))
    xtick = [str(r["test_start"]) for _, r in df.iterrows()]

    # ── 1. Accuracy ──
    ax = axes[0]
    colors = ["#00d4aa" if v > 52 else "#ff6b6b" for v in df["accuracy"]]
    ax.bar(x, df["accuracy"], color=colors, alpha=0.85, width=0.6)
    ax.axhline(52, color="#ffd700", ls="--", lw=1.5, label="%52 baseline")
    ax.axhline(df["accuracy"].mean(), color="#fff", ls=":", lw=1,
               label=f"Ort. {df['accuracy'].mean():.1f}%")
    ax.set_ylabel("Accuracy (%)", color="#8b9cad")
    ax.set_ylim(45, 65)
    ax.legend(fontsize=8, facecolor="#1e2432", labelcolor="white", framealpha=0.8)
    ax.set_title("Walk-Forward Backtest — Binary Model (UP/DOWN)", color="white",
                 fontsize=13, pad=10)

    # ── 2. Precision ──
    ax = axes[1]
    ax.plot(x, df["up_prec"], color="#00d4aa", lw=2, marker="o", ms=5, label="UP Precision")
    ax.plot(x, df["dn_prec"], color="#ff6b6b", lw=2, marker="s", ms=5, label="DOWN Precision")
    ax.axhline(50, color="#555", ls=":", lw=1)
    ax.axhline(df["up_prec"].mean(), color="#00d4aa", ls="--", lw=1, alpha=0.5)
    ax.axhline(df["dn_prec"].mean(), color="#ff6b6b", ls="--", lw=1, alpha=0.5)
    ax.set_ylabel("Precision (%)", color="#8b9cad")
    ax.set_ylim(40, 70)
    ax.legend(fontsize=8, facecolor="#1e2432", labelcolor="white", framealpha=0.8)

    # ── 3. DOWN bias ──
    ax = axes[2]
    ax.fill_between(x, df["bias_ratio"], alpha=0.3, color="#f39c12")
    ax.plot(x, df["bias_ratio"], color="#f39c12", lw=2, marker="D", ms=5, label="DOWN/UP sinyal orani")
    ax.axhline(1.0, color="#fff", ls="--", lw=1, label="Ideal (1.0x)")
    ax.set_ylabel("DOWN bias (x)", color="#8b9cad")
    ax.set_xlabel("Test donemi", color="#8b9cad")
    ax.legend(fontsize=8, facecolor="#1e2432", labelcolor="white", framealpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(xtick, rotation=35, ha="right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path("data/walk_forward.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[+] Grafik: {out}")


def main():
    print("=" * 55)
    print("  WALK-FORWARD BACKTEST")
    print(f"  Train={TRAIN_MONTHS}ay  Test={TEST_MONTHS}ay  Adim={STEP_MONTHS}ay")
    print("=" * 55)

    df     = load_data()
    feat   = build_features(df)
    target = build_target(df)

    combined = feat.copy()
    combined["__target__"] = target
    combined.dropna(inplace=True)
    feature_cols = get_feature_cols(combined)
    print(f"[+] Feature sayisi: {len(feature_cols)}")

    folds   = make_folds(combined)
    results = []

    print(f"\n[*] Foldlar isleniyor...\n")
    for i, fold in enumerate(folds):
        r = run_fold(combined, combined["__target__"], feature_cols, fold, i)
        if r:
            results.append(r)

    if not results:
        print("[HATA] Hic sonuc uretilmedi — veri yeterli degil.")
        return

    print_summary(results)
    plot_results(results)

    out_csv = Path("data/walk_forward.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[+] CSV: {out_csv}")

    print(f"\n{'='*55}")
    print(f"  SONRAKI ADIM: class weight dengesi")
    print(f"  DOWN bias tutarliysa -> ml_model.py'da")
    print(f"  CLASS_WEIGHTS duzelt, modeli yeniden egit")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
