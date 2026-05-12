"""
ORDERFLOW BOT - Threshold Sweep (ml_model.py eklentisi)
=========================================================
Kullanim:
    python threshold_sweep.py

Aciklama:
    Egitilmis xgb_binary.pkl modelini yukler.
    Test seti uzerinde 0.50-0.75 arasi her 0.01 esigini dener.
    Her esik icin: UP/DOWN precision, coverage, EV skoru hesaplar.
    En iyi esigi onerir. Grafik: data/threshold_sweep.png
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI gerektirmez, PNG kaydeder
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ─── AYARLAR (ml_model.py ile ayni olmali) ──────────────────────────────────
PARQUET_PATH     = Path("data/ETHUSDT_orderflow_365d_5m.parquet")
MODEL_B_PATH     = Path("data/xgb_binary.pkl")
TRAIN_RATIO      = 0.70
TARGET_BARS      = 3
TARGET_THRESHOLD = 0.0015

SWEEP_MIN  = 0.50
SWEEP_MAX  = 0.75
SWEEP_STEP = 0.01
# ─────────────────────────────────────────────────────────────────────────────


def load_model(path):
    with open(path, "rb") as f:
        p = pickle.load(f)
    return p["model"], p["le"], p["feature_cols"]


def load_data():
    df = pd.read_parquet(PARQUET_PATH)
    df.sort_index(inplace=True)
    return df


def build_features(df):
    """ml_model.py'daki build_features ile ayni."""
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


def prepare_test_set(df, feature_cols):
    feat   = build_features(df)
    target = build_target(df)

    combined = feat.copy()
    combined["__target__"] = target
    combined.dropna(inplace=True)

    X = combined[feature_cols]
    y = combined["__target__"]

    split_idx = int(len(X) * TRAIN_RATIO)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # Binary: sadece UP/DOWN (FLAT cikart)
    mask   = y_test != 0
    X_test = X_test[mask]
    y_test = y_test[mask]

    print(f"[+] Test seti: {len(X_test):,} bar (FLAT cikarildi)")
    print(f"    UP: {(y_test==1).sum():,}  |  DOWN: {(y_test==-1).sum():,}")
    return X_test, y_test


def run_sweep(model, le, X_test, y_test):
    """
    Her esik icin:
      - UP precision  : model UP dediginde gercekten UP olma orani
      - DOWN precision: model DOWN dediginde gercekten DOWN olma orani
      - coverage      : esigi gecen bar sayisi / toplam bar
      - ev_score      : (up_prec + down_prec) / 2 * coverage  (genel kalite)
    """
    # Tum barlar icin ham olasiliklar
    probs  = model.predict_proba(X_test)          # shape (n, 2)
    # le.classes_: genellikle [-1, 1] -> [0, 1] encode
    # DOWN=le.classes_[0], UP=le.classes_[1]
    classes = le.classes_   # [-1, 1]
    down_idx = list(classes).index(-1)
    up_idx   = list(classes).index(1)

    prob_up   = probs[:, up_idx]
    prob_down = probs[:, down_idx]

    y_arr = y_test.values
    total = len(y_arr)

    thresholds = np.arange(SWEEP_MIN, SWEEP_MAX + SWEEP_STEP/2, SWEEP_STEP)
    results = []

    for thr in thresholds:
        # Model UP diyor: prob_up >= thr
        up_mask    = prob_up >= thr
        up_correct = (y_arr[up_mask] == 1).sum()
        up_total   = up_mask.sum()
        up_prec    = up_correct / up_total if up_total > 0 else 0.0

        # Model DOWN diyor: prob_down >= thr
        dn_mask    = prob_down >= thr
        dn_correct = (y_arr[dn_mask] == -1).sum()
        dn_total   = dn_mask.sum()
        dn_prec    = dn_correct / dn_total if dn_total > 0 else 0.0

        # Coverage: esigi gecen bar sayisi
        covered  = (up_mask | dn_mask).sum()
        coverage = covered / total

        # EV skoru: precision kalitesi * ne kadar sinyal urettigimiz
        ev = (up_prec + dn_prec) / 2 * coverage

        results.append({
            "threshold"   : round(thr, 2),
            "up_prec"     : up_prec,
            "dn_prec"     : dn_prec,
            "avg_prec"    : (up_prec + dn_prec) / 2,
            "coverage"    : coverage,
            "up_signals"  : int(up_total),
            "dn_signals"  : int(dn_total),
            "ev_score"    : ev,
        })

    return pd.DataFrame(results)


def print_table(df):
    print(f"\n{'='*80}")
    print(f"  THRESHOLD SWEEP SONUCLARI")
    print(f"{'='*80}")
    print(f"  {'Thr':>5} {'UP Prec':>9} {'DN Prec':>9} {'Avg Prec':>10} "
          f"{'Coverage':>10} {'UP Sig':>8} {'DN Sig':>8} {'EV':>8}")
    print(f"  {'-'*75}")

    best_ev  = df["ev_score"].idxmax()
    best_prec = df["avg_prec"].idxmax()

    for i, row in df.iterrows():
        marker = ""
        if i == best_ev:   marker += " ← MAX EV"
        if i == best_prec: marker += " ← MAX PREC"
        print(f"  {row['threshold']:>5.2f} "
              f"{row['up_prec']*100:>8.1f}% "
              f"{row['dn_prec']*100:>8.1f}% "
              f"{row['avg_prec']*100:>9.1f}% "
              f"{row['coverage']*100:>9.1f}% "
              f"{row['up_signals']:>8,} "
              f"{row['dn_signals']:>8,} "
              f"{row['ev_score']*100:>7.2f}%"
              f"{marker}")

    print(f"{'='*80}")

    best = df.loc[best_ev]
    print(f"\n  ✓ EN IYI EV ESIGI      : {best['threshold']:.2f}")
    print(f"    UP Precision         : {best['up_prec']*100:.1f}%")
    print(f"    DOWN Precision       : {best['dn_prec']*100:.1f}%")
    print(f"    Coverage             : {best['coverage']*100:.1f}% ({best['up_signals']+best['dn_signals']:,} sinyal)")
    print(f"\n  Mevcut esik (0.60):")
    cur = df[df["threshold"] == 0.60].iloc[0]
    print(f"    UP Precision         : {cur['up_prec']*100:.1f}%")
    print(f"    DOWN Precision       : {cur['dn_prec']*100:.1f}%")
    print(f"    Coverage             : {cur['coverage']*100:.1f}%")
    print(f"    EV Skoru             : {cur['ev_score']*100:.2f}%")


def plot_sweep(df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b9cad")
        ax.spines[:].set_color("#2d3748")

    x = df["threshold"]

    # ── 1. Precision ──
    ax = axes[0]
    ax.plot(x, df["up_prec"]*100,  color="#00d4aa", lw=2, label="UP Precision")
    ax.plot(x, df["dn_prec"]*100,  color="#ff6b6b", lw=2, label="DOWN Precision")
    ax.plot(x, df["avg_prec"]*100, color="#ffd700", lw=1.5, ls="--", label="Avg Precision")
    ax.axhline(50, color="#555", ls=":", lw=1)
    ax.axvline(0.60, color="#888", ls="--", lw=1, label="Mevcut (0.60)")
    best_ev_thr = df.loc[df["ev_score"].idxmax(), "threshold"]
    ax.axvline(best_ev_thr, color="#a78bfa", ls="--", lw=1.5, label=f"Best EV ({best_ev_thr:.2f})")
    ax.set_ylabel("Precision (%)", color="#8b9cad")
    ax.legend(fontsize=8, facecolor="#1e2432", labelcolor="white", framealpha=0.8)
    ax.set_title("Threshold Sweep — Binary Model (UP/DOWN)", color="white", fontsize=13, pad=10)

    # ── 2. Coverage ──
    ax = axes[1]
    ax.fill_between(x, df["coverage"]*100, alpha=0.3, color="#60a5fa")
    ax.plot(x, df["coverage"]*100, color="#60a5fa", lw=2, label="Coverage")
    ax.axvline(0.60, color="#888", ls="--", lw=1)
    ax.axvline(best_ev_thr, color="#a78bfa", ls="--", lw=1.5)
    ax.set_ylabel("Coverage (%)", color="#8b9cad")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=8, facecolor="#1e2432", labelcolor="white", framealpha=0.8)

    # ── 3. EV Score ──
    ax = axes[2]
    ax.fill_between(x, df["ev_score"]*100, alpha=0.3, color="#a78bfa")
    ax.plot(x, df["ev_score"]*100, color="#a78bfa", lw=2, label="EV Score")
    best_row = df.loc[df["ev_score"].idxmax()]
    ax.scatter([best_row["threshold"]], [best_row["ev_score"]*100],
               color="#ffd700", s=80, zorder=5, label=f"Max EV @ {best_row['threshold']:.2f}")
    ax.axvline(0.60, color="#888", ls="--", lw=1, label="Mevcut (0.60)")
    ax.set_xlabel("Confidence Threshold", color="#8b9cad")
    ax.set_ylabel("EV Score (%)", color="#8b9cad")
    ax.legend(fontsize=8, facecolor="#1e2432", labelcolor="white", framealpha=0.8)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path("data/threshold_sweep.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[+] Grafik kaydedildi: {out}")


def main():
    print("[*] Threshold Sweep basliyor...")
    print(f"[*] Model: {MODEL_B_PATH}")
    print(f"[*] Aralik: {SWEEP_MIN:.2f} - {SWEEP_MAX:.2f} (adim: {SWEEP_STEP})")

    model, le, feature_cols = load_model(MODEL_B_PATH)
    df = load_data()
    print(f"[+] Veri: {len(df):,} bar")

    X_test, y_test = prepare_test_set(df, feature_cols)
    results_df     = run_sweep(model, le, X_test, y_test)

    print_table(results_df)
    plot_sweep(results_df)

    # CSV kaydet
    out_csv = Path("data/threshold_sweep.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"[+] CSV kaydedildi: {out_csv}")

    # Tavsiye
    best = results_df.loc[results_df["ev_score"].idxmax()]
    cur  = results_df[results_df["threshold"] == 0.60].iloc[0]
    print(f"\n{'='*55}")
    print(f"  TAVSIYE")
    print(f"{'='*55}")
    if best["threshold"] != 0.60:
        delta_prec = (best["avg_prec"] - cur["avg_prec"]) * 100
        delta_cov  = (best["coverage"] - cur["coverage"]) * 100
        print(f"  ML_CONF_THRESHOLD = {best['threshold']:.2f}  (su an: 0.60)")
        print(f"  Precision farki: {delta_prec:+.1f}pp")
        print(f"  Coverage farki : {delta_cov:+.1f}pp")
        print(f"\n  signal_logger.py'da degistir:")
        print(f"  ML_CONF_THRESHOLD = {best['threshold']:.2f}")
    else:
        print(f"  Mevcut esik (0.60) zaten optimal!")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()