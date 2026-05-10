"""
signal_engine_v2.py
===================
ORDERFLOW BOT — v2 Engine
- ICT tamamen kaldırıldı
- Orderflow bileşenleri genişletildi + detay analizi
- Absorption dedektörü (Bullish & Bearish)
- Dynamic Exit sistemi (3 katman: CVD Reversal, ATR Trailing, BPR/IFVG)
- signal_logger.py tarafından import edilir
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from colorama import Fore, Style, init

init(autoreset=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CFG = {
    # Orderflow eşikleri
    "imbalance_bull":         0.58,
    "imbalance_bear":         0.42,
    "volume_spike_mult":      1.5,
    "cvd_lookback":           5,
    "delta_ma_fast":          3,
    "delta_ma_slow":          10,
    "session_lookback":       12,       # session delta için bar sayısı

    # Absorption
    "absorption_vol_mult":    1.5,      # volume > avg20 * 1.5
    "absorption_body_atr":    1.0,      # gövde < ATR * 1.0
    "absorption_delta_pct":   0.30,     # |delta| / volume > %30 → güçlü basınç
    "absorption_new_extreme_bars": 5,   # son N barda yeni dip/zirve yok mu?

    # Dynamic Exit
    "atr_trail_mult":         2.5,      # ATR trailing stop çarpanı
    "cvd_div_lookback":       10,       # CVD divergence için bakış penceresi
    "bpr_ifvg_lookback":      30,       # BPR/IFVG zone için bakış penceresi
    "bpr_min_overlap":        0.0003,   # min overlap oranı (%0.03)

    # Skorlama
    "min_score_long":         6.0,
    "min_score_short":        6.0,

    # Risk
    "balance":                2500.0,
    "atr_sl_mult":            1.5,
    "sl_min_pct":             0.003,
    "tp1_r":                  1.5,
    "tp2_r":                  3.0,
    "tp3_r":                  5.0,
    "lev_min":                2,
    "lev_max":                15,
}

# ─────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────
@dataclass
class FundingOIResult:
    # Funding Rate
    funding_rate: float = 0.0          # mevcut funding rate (örn: 0.0001)
    funding_rate_pct: float = 0.0      # yüzde olarak (örn: 0.01%)
    funding_bias: str = ""             # "LONG_BIASED" | "SHORT_BIASED" | "NEUTRAL"
    # Open Interest
    oi_current: float = 0.0
    oi_prev: float = 0.0
    oi_change_pct: float = 0.0         # OI değişim yüzdesi
    oi_trend: str = ""                 # "RISING" | "FALLING" | "FLAT"
    # Skorlar
    score_funding: float = 0.0        # -0.75 ile +0.75
    score_oi: float = 0.0             # -0.75 ile +0.75
    detail: str = ""


@dataclass
class AbsorptionResult:
    bullish: bool = False
    bearish: bool = False
    score_bonus: float = 0.0
    volume_ratio: float = 0.0        # volume / avg20
    body_vs_atr: float = 0.0         # gövde / ATR
    delta_pressure: float = 0.0      # |delta| / volume
    new_extreme: bool = False        # son N barda yeni dip/zirve var mı?
    detail: str = ""


@dataclass
class DynamicExitResult:
    should_exit: bool = False
    reason: str = ""
    # CVD Divergence
    cvd_divergence: bool = False
    cvd_div_type: str = ""           # "bearish_div" | "bullish_div"
    # ATR Trailing
    atr_trail_long: float = 0.0      # long pozisyon için trail seviyesi
    atr_trail_short: float = 0.0     # short pozisyon için trail seviyesi
    atr_value: float = 0.0
    # BPR/IFVG
    bpr_zone_hit: bool = False
    bpr_zone_top: float = 0.0
    bpr_zone_bot: float = 0.0
    detail: str = ""


@dataclass
class OrderflowDetail:
    # Ham değerler
    delta: float = 0.0
    delta_pct: float = 0.0           # delta / volume
    cvd: float = 0.0
    cvd_slope: float = 0.0           # son N bar CVD değişimi
    imbalance_ratio: float = 0.0
    stacked_up: bool = False
    stacked_dn: bool = False
    volume: float = 0.0
    volume_avg20: float = 0.0
    volume_ratio: float = 0.0        # volume / avg20
    session_delta: float = 0.0
    session_delta_direction: str = ""
    vwap: float = 0.0
    price_vs_vwap: str = ""          # "above" | "below" | "at"
    bar_close_pos: float = 0.0       # (close-low)/(high-low) → 0-1
    # Skorlar (her bileşen ayrı)
    score_delta:        float = 0.0
    score_cvd:          float = 0.0
    score_imbalance:    float = 0.0
    score_stacked:      float = 0.0
    score_session:      float = 0.0
    score_vwap:         float = 0.0
    score_vol_spike:    float = 0.0
    score_bar_close:    float = 0.0
    score_delta_ma:     float = 0.0
    score_absorption:   float = 0.0
    score_funding:      float = 0.0
    score_oi:           float = 0.0
    total_score:        float = 0.0
    direction: str = ""              # "LONG" | "SHORT" | "FLAT"


@dataclass
class SignalResult:
    symbol: str = "ETHUSDT"
    timestamp: str = ""
    direction: str = "FLAT"          # "LONG" | "SHORT" | "FLAT"
    score: float = 0.0
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    leverage: int = 1
    atr: float = 0.0
    orderflow: OrderflowDetail = field(default_factory=OrderflowDetail)
    absorption: AbsorptionResult = field(default_factory=AbsorptionResult)
    dynamic_exit: DynamicExitResult = field(default_factory=DynamicExitResult)
    funding_oi: FundingOIResult = field(default_factory=FundingOIResult)
    exit_signal: bool = False
    exit_reason: str = ""


# ─────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────
def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True Range tabanlı ATR."""
    high = df["high"]
    low  = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _find_bpr_ifvg_zones(df: pd.DataFrame, lookback: int = 30) -> list[dict]:
    """
    BPR (Balanced Price Range) / IFVG (Inverse Fair Value Gap) zone'larını tespit eder.
    Sadece dynamic exit için kullanılır — sinyal üretmez.

    BPR: iki zıt yönlü FVG üst üste örtüşüyorsa → örtüşen alan BPR zone
    IFVG: dolu FVG'nin içine tekrar girilmişse → zone hâlâ aktif

    Basitleştirilmiş yaklaşım:
    - Bullish FVG: bar[i-2].high < bar[i].low → zone: [bar[i-2].high, bar[i].low]
    - Bearish FVG: bar[i-2].low  > bar[i].high→ zone: [bar[i].high, bar[i-2].low]
    - BPR: bull FVG zone ∩ bear FVG zone → overlap varsa BPR
    """
    window = df.iloc[-lookback:].reset_index(drop=True)
    bull_zones = []
    bear_zones = []

    for i in range(2, len(window)):
        # Bullish FVG
        if window["high"].iloc[i-2] < window["low"].iloc[i]:
            bull_zones.append({
                "top": window["low"].iloc[i],
                "bot": window["high"].iloc[i-2],
                "bar": i
            })
        # Bearish FVG
        if window["low"].iloc[i-2] > window["high"].iloc[i]:
            bear_zones.append({
                "top": window["low"].iloc[i-2],
                "bot": window["high"].iloc[i],
                "bar": i
            })

    # BPR = bull_zone ∩ bear_zone overlap
    bpr_zones = []
    for bz in bull_zones:
        for sz in bear_zones:
            overlap_top = min(bz["top"], sz["top"])
            overlap_bot = max(bz["bot"], sz["bot"])
            if overlap_top > overlap_bot:
                bpr_zones.append({"top": overlap_top, "bot": overlap_bot, "type": "BPR"})

    # IFVG: bull/bear zone'larını da döndür (BPR yoksa bunlara bakılır)
    all_zones = bpr_zones + \
                [{"top": z["top"], "bot": z["bot"], "type": "bull_ifvg"} for z in bull_zones[-3:]] + \
                [{"top": z["top"], "bot": z["bot"], "type": "bear_ifvg"} for z in bear_zones[-3:]]
    return all_zones


# ─────────────────────────────────────────────
# FUNDING RATE + OI SCORER
# ─────────────────────────────────────────────
def score_funding_oi(
    funding_rate: float,   # ham funding rate, örn: 0.0001
    oi_current: float,     # mevcut OI
    oi_prev: float,        # önceki OI (1 periyot önce)
    price_direction: float # delta yönü: >0 long, <0 short
) -> FundingOIResult:
    """
    Funding Rate + OI kombinasyon analizi.

    Funding Rate Mantığı:
      - Pozitif funding (longlar ödüyor) → piyasa aşırı long biased
        * Fiyat yükseliyorsa → LONG güçlenir (+0.5) — trend devam
        * Fiyat düşüyorsa   → SHORT güçlenir (+0.75) — long sıkışması
      - Negatif funding (shortlar ödüyor) → piyasa aşırı short biased
        * Fiyat düşüyorsa   → SHORT güçlenir (+0.5) — trend devam
        * Fiyat yükseliyorsa→ LONG güçlenir (+0.75) — short sıkışması

    OI Mantığı:
      - OI artıyor + delta pozitif  → yeni long pozisyon açılıyor (+0.75)
      - OI artıyor + delta negatif  → yeni short pozisyon açılıyor (-0.75)
      - OI azalıyor + fiyat düşüyor → long kapanıyor (zayıflama, -0.5)
      - OI azalıyor + fiyat yükseliyor → short kapanıyor (zayıflama, +0.5)
    """
    res = FundingOIResult()
    res.funding_rate     = funding_rate
    res.funding_rate_pct = round(funding_rate * 100, 4)
    res.oi_current       = oi_current
    res.oi_prev          = oi_prev

    # OI değişimi
    if oi_prev > 0:
        res.oi_change_pct = round((oi_current - oi_prev) / oi_prev * 100, 3)
    oi_rising  = res.oi_change_pct > 0.1   # %0.1 üzeri artış
    oi_falling = res.oi_change_pct < -0.1  # %0.1 üzeri düşüş
    res.oi_trend = "RISING" if oi_rising else ("FALLING" if oi_falling else "FLAT")

    # ── FUNDING RATE SKORU ────────────────────────────────────────
    fr_threshold = 0.0001  # %0.01 eşiği

    if funding_rate > fr_threshold:
        # Pozitif funding → longlar ödüyor
        res.funding_bias = "LONG_BIASED"
        if price_direction > 0:
            res.score_funding = +0.5   # trend devam
        else:
            res.score_funding = +0.75  # long sıkışması → SHORT güçlü
            res.score_funding *= -1    # short yönünde skor

    elif funding_rate < -fr_threshold:
        # Negatif funding → shortlar ödüyor
        res.funding_bias = "SHORT_BIASED"
        if price_direction < 0:
            res.score_funding = -0.5   # trend devam (short yönünde)
        else:
            res.score_funding = +0.75  # short sıkışması → LONG güçlü
    else:
        res.funding_bias  = "NEUTRAL"
        res.score_funding = 0.0

    # ── OI SKORU ─────────────────────────────────────────────────
    if oi_rising:
        # Yeni pozisyon açılıyor
        if price_direction > 0:
            res.score_oi = +0.75   # yeni long → LONG güçlü
        else:
            res.score_oi = -0.75   # yeni short → SHORT güçlü
    elif oi_falling:
        # Pozisyon kapanıyor
        if price_direction > 0:
            res.score_oi = +0.5    # short kapanıyor → LONG hafif güçlü
        else:
            res.score_oi = -0.5    # long kapanıyor → SHORT hafif güçlü
    else:
        res.score_oi = 0.0

    # ── DETAY ────────────────────────────────────────────────────
    res.detail = (
        f"Funding: %{res.funding_rate_pct:.4f} ({res.funding_bias}) → skor:{res.score_funding:+.2f} | "
        f"OI: {res.oi_trend} ({res.oi_change_pct:+.2f}%) → skor:{res.score_oi:+.2f}"
    )
    return res


# ─────────────────────────────────────────────
# ABSORPTION DETECTOR
# ─────────────────────────────────────────────
def detect_absorption(df: pd.DataFrame) -> AbsorptionResult:
    """
    Bullish Absorption:
      - Volume > avg20 * 1.5  (olağandışı hacim)
      - |open - close| < ATR  (küçük gövde, piyasa bir yere gitmiyor)
      - delta < 0 VE |delta|/volume > eşik  (güçlü satış baskısı var)
      - Son N barda yeni dip yapılmıyor  (alıcı absorbe ediyor)

    Bearish Absorption: tersi
    """
    res = AbsorptionResult()
    if len(df) < 22:
        return res

    cur     = df.iloc[-1]
    atr_val = _atr(df).iloc[-1]
    if pd.isna(atr_val) or atr_val == 0:
        return res

    vol_avg20    = df["volume"].iloc[-21:-1].mean()
    body         = abs(cur["open"] - cur["close"])
    delta        = cur.get("delta", 0.0)
    volume       = cur["volume"]
    n            = CFG["absorption_new_extreme_bars"]
    recent_low   = df["low"].iloc[-n-1:-1].min()
    recent_high  = df["high"].iloc[-n-1:-1].max()

    vol_ratio    = volume / vol_avg20 if vol_avg20 > 0 else 0
    body_ratio   = body / atr_val
    delta_pres   = abs(delta) / volume if volume > 0 else 0
    new_low      = cur["low"] < recent_low
    new_high     = cur["high"] > recent_high

    res.volume_ratio    = round(vol_ratio, 3)
    res.body_vs_atr     = round(body_ratio, 3)
    res.delta_pressure  = round(delta_pres, 3)

    high_vol   = vol_ratio   >= CFG["absorption_vol_mult"]
    small_body = body_ratio  <  CFG["absorption_body_atr"]
    strong_pressure = delta_pres >= CFG["absorption_delta_pct"]

    # Bullish: satış baskısı var (delta negatif) ama yeni dip yok
    if high_vol and small_body and strong_pressure and delta < 0 and not new_low:
        res.bullish   = True
        res.new_extreme = False
        res.score_bonus = 2.0
        res.detail = (
            f"🐋 Bullish Absorption | "
            f"Vol {vol_ratio:.2f}x avg | "
            f"Gövde ATR'nin %{body_ratio*100:.0f}'i | "
            f"Delta baskısı %{delta_pres*100:.0f} | "
            f"Yeni dip YOK → Limit alıcı var"
        )

    # Bearish: alış baskısı var (delta pozitif) ama yeni zirve yok
    elif high_vol and small_body and strong_pressure and delta > 0 and not new_high:
        res.bearish   = True
        res.new_extreme = False
        res.score_bonus = 2.0
        res.detail = (
            f"🐋 Bearish Absorption | "
            f"Vol {vol_ratio:.2f}x avg | "
            f"Gövde ATR'nin %{body_ratio*100:.0f}'i | "
            f"Delta baskısı %{delta_pres*100:.0f} | "
            f"Yeni zirve YOK → Limit satıcı var"
        )
    else:
        res.detail = (
            f"Absorption YOK | "
            f"Vol {vol_ratio:.2f}x | "
            f"Gövde %{body_ratio*100:.0f} ATR | "
            f"Delta baskı %{delta_pres*100:.0f}"
        )

    return res


# ─────────────────────────────────────────────
# DYNAMIC EXIT
# ─────────────────────────────────────────────
def compute_dynamic_exit(
    df: pd.DataFrame,
    position: str,          # "LONG" | "SHORT"
    entry_price: float,
) -> DynamicExitResult:
    """
    3 katmanlı dynamic exit:
    1. CVD Divergence  — fiyat yeni HH/LL yaparken CVD yapmıyorsa
    2. ATR Trailing    — 2.5 * ATR mesafesiyle trailing stop
    3. BPR/IFVG İhlali — zone'a temas veya ihlal
    """
    res = DynamicExitResult()
    if len(df) < 20:
        return res

    atr_val = _atr(df).iloc[-1]
    if pd.isna(atr_val):
        return res
    res.atr_value = round(atr_val, 4)

    cur_close = df["close"].iloc[-1]
    cur_high  = df["high"].iloc[-1]
    cur_low   = df["low"].iloc[-1]

    # ── 1. CVD DIVERGENCE ──────────────────────────────────────────────
    lookback = CFG["cvd_div_lookback"]
    window   = df.iloc[-lookback:]
    price_col = "close"
    cvd_col   = "cvd"

    if cvd_col in df.columns:
        price_hh = window[price_col].iloc[-1] > window[price_col].iloc[:-1].max()
        price_ll = window[price_col].iloc[-1] < window[price_col].iloc[:-1].min()
        cvd_hh   = window[cvd_col].iloc[-1]   > window[cvd_col].iloc[:-1].max()
        cvd_ll   = window[cvd_col].iloc[-1]   < window[cvd_col].iloc[:-1].min()

        # Bearish divergence: fiyat yeni HH ama CVD yapmıyor → LONG çık
        if position == "LONG" and price_hh and not cvd_hh:
            res.cvd_divergence = True
            res.cvd_div_type   = "bearish_div"
            res.should_exit    = True
            res.reason         = "CVD Bearish Divergence (fiyat HH, CVD hayır)"

        # Bullish divergence: fiyat yeni LL ama CVD yapmıyor → SHORT çık
        elif position == "SHORT" and price_ll and not cvd_ll:
            res.cvd_divergence = True
            res.cvd_div_type   = "bullish_div"
            res.should_exit    = True
            res.reason         = "CVD Bullish Divergence (fiyat LL, CVD hayır)"

    # ── 2. ATR TRAILING STOP ───────────────────────────────────────────
    trail_dist = CFG["atr_trail_mult"] * atr_val
    res.atr_trail_long  = round(cur_high - trail_dist, 4)
    res.atr_trail_short = round(cur_low  + trail_dist, 4)

    if position == "LONG" and cur_close < res.atr_trail_long:
        if not res.should_exit:
            res.should_exit = True
            res.reason      = f"ATR Trail kırıldı (trail: {res.atr_trail_long:.2f})"
    elif position == "SHORT" and cur_close > res.atr_trail_short:
        if not res.should_exit:
            res.should_exit = True
            res.reason      = f"ATR Trail kırıldı (trail: {res.atr_trail_short:.2f})"

    # ── 3. BPR / IFVG ZONE İHLALİ ─────────────────────────────────────
    zones = _find_bpr_ifvg_zones(df, CFG["bpr_ifvg_lookback"])
    for zone in zones:
        top = zone["top"]
        bot = zone["bot"]
        ztype = zone["type"]

        # Fiyat zone içine girdi mi?
        in_zone = bot <= cur_close <= top

        if in_zone:
            res.bpr_zone_hit = True
            res.bpr_zone_top = top
            res.bpr_zone_bot = bot

            # LONG pozisyon: bearish zone'a (bear_ifvg veya BPR) girerse çık
            if position == "LONG" and ztype in ("bear_ifvg", "BPR"):
                if not res.should_exit:
                    res.should_exit = True
                    res.reason      = f"{ztype} zone ihlali ({bot:.2f}-{top:.2f})"
            # SHORT pozisyon: bullish zone'a girerse çık
            elif position == "SHORT" and ztype in ("bull_ifvg", "BPR"):
                if not res.should_exit:
                    res.should_exit = True
                    res.reason      = f"{ztype} zone ihlali ({bot:.2f}-{top:.2f})"

    # ── DETAY METNİ ───────────────────────────────────────────────────
    exit_reasons = []
    if res.cvd_divergence:
        exit_reasons.append(f"CVD Div({res.cvd_div_type})")
    exit_reasons.append(
        f"ATR Trail → LONG:{res.atr_trail_long:.2f} / SHORT:{res.atr_trail_short:.2f}"
    )
    if res.bpr_zone_hit:
        exit_reasons.append(f"BPR/IFVG zone [{res.bpr_zone_bot:.2f}-{res.bpr_zone_top:.2f}]")

    res.detail = " | ".join(exit_reasons) if exit_reasons else "Exit tetikleyici yok"
    return res


# ─────────────────────────────────────────────
# ORDERFLOW SCORER
# ─────────────────────────────────────────────
def score_orderflow(df: pd.DataFrame) -> OrderflowDetail:
    """
    9 bileşen + Absorption bonus → detaylı orderflow skoru.

    Bileşenler:
      1. Delta yönü          → 0–1.5
      2. CVD momentum        → 0–1.0
      3. Imbalance ratio     → 0–1.5
      4. Stacked imbalance   → 0–2.0
      5. Session delta       → 0–0.75
      6. Price vs VWAP       → 0–0.5
      7. Volume spike+delta  → 0–1.0
      8. Bar kapanış pozisyonu → 0–0.5
      9. Delta MA3 vs MA10   → 0–0.5
    +10. Absorption bonus    → 0–2.0
    """
    od = OrderflowDetail()
    if len(df) < 20:
        return od

    cur = df.iloc[-1]

    # Ham değerler
    od.delta          = cur.get("delta", 0.0)
    od.volume         = cur.get("volume", 0.0)
    od.delta_pct      = od.delta / od.volume if od.volume > 0 else 0.0
    od.imbalance_ratio = cur.get("imbalance_ratio", 0.5)
    od.stacked_up     = bool(cur.get("stacked_imbalance_up", False))
    od.stacked_dn     = bool(cur.get("stacked_imbalance_dn", False))
    od.vwap           = cur.get("vwap", cur["close"])
    od.volume_avg20   = df["volume"].iloc[-21:-1].mean()
    od.volume_ratio   = od.volume / od.volume_avg20 if od.volume_avg20 > 0 else 1.0

    # CVD slope (son N bar)
    n = CFG["cvd_lookback"]
    if "cvd" in df.columns and len(df) >= n + 1:
        od.cvd       = cur.get("cvd", 0.0)
        od.cvd_slope = df["cvd"].iloc[-1] - df["cvd"].iloc[-n]
    
    # Session delta
    s_lb = CFG["session_lookback"]
    if "session_delta" in df.columns:
        od.session_delta = cur.get("session_delta", 0.0)
    else:
        od.session_delta = df["delta"].iloc[-s_lb:].sum() if "delta" in df.columns else 0.0

    od.session_delta_direction = "BULL" if od.session_delta > 0 else "BEAR"

    # Price vs VWAP
    close = cur["close"]
    vwap_band = od.vwap * 0.0005
    if close > od.vwap + vwap_band:
        od.price_vs_vwap = "above"
    elif close < od.vwap - vwap_band:
        od.price_vs_vwap = "below"
    else:
        od.price_vs_vwap = "at"

    # Bar kapanış pozisyonu (0=dip, 1=tepe)
    hl_range = cur["high"] - cur["low"]
    od.bar_close_pos = (close - cur["low"]) / hl_range if hl_range > 0 else 0.5

    # Delta MA
    delta_ma_fast = CFG["delta_ma_fast"]
    delta_ma_slow = CFG["delta_ma_slow"]
    if "delta" in df.columns and len(df) >= delta_ma_slow:
        dma_fast = df["delta"].iloc[-delta_ma_fast:].mean()
        dma_slow = df["delta"].iloc[-delta_ma_slow:].mean()
    else:
        dma_fast = dma_slow = 0.0

    # ── SKORLAMA ──────────────────────────────────────────────────────

    # 1. Delta yönü
    if od.delta > 0:
        od.score_delta = min(1.5, 1.5 * min(od.delta_pct / 0.1, 1.0))
    elif od.delta < 0:
        od.score_delta = -min(1.5, 1.5 * min(abs(od.delta_pct) / 0.1, 1.0))

    # 2. CVD momentum
    if od.cvd_slope > 0:
        od.score_cvd = 1.0
    elif od.cvd_slope < 0:
        od.score_cvd = -1.0

    # 3. Imbalance ratio
    if od.imbalance_ratio >= CFG["imbalance_bull"]:
        od.score_imbalance = 1.5
    elif od.imbalance_ratio <= CFG["imbalance_bear"]:
        od.score_imbalance = -1.5

    # 4. Stacked imbalance
    if od.stacked_up:
        od.score_stacked = 2.0
    elif od.stacked_dn:
        od.score_stacked = -2.0

    # 5. Session delta
    if od.session_delta > 0:
        od.score_session = 0.75
    elif od.session_delta < 0:
        od.score_session = -0.75

    # 6. VWAP
    if od.price_vs_vwap == "above":
        od.score_vwap = 0.5
    elif od.price_vs_vwap == "below":
        od.score_vwap = -0.5

    # 7. Volume spike + delta konfirmasyonu
    if od.volume_ratio >= CFG["volume_spike_mult"]:
        if od.delta > 0:
            od.score_vol_spike = 1.0
        elif od.delta < 0:
            od.score_vol_spike = -1.0

    # 8. Bar kapanış pozisyonu
    if od.bar_close_pos >= 0.7:
        od.score_bar_close = 0.5   # güçlü kapanış yukarda
    elif od.bar_close_pos <= 0.3:
        od.score_bar_close = -0.5  # güçlü kapanış aşağıda

    # 9. Delta MA3 vs MA10
    if dma_fast > dma_slow:
        od.score_delta_ma = 0.5
    elif dma_fast < dma_slow:
        od.score_delta_ma = -0.5

    # Ham toplam (absorption henüz yok)
    raw_score = (
        od.score_delta +
        od.score_cvd +
        od.score_imbalance +
        od.score_stacked +
        od.score_session +
        od.score_vwap +
        od.score_vol_spike +
        od.score_bar_close +
        od.score_delta_ma
    )

    # Absorption bonus (dışarıdan eklenir, burada placeholder)
    od.total_score = round(raw_score, 3)
    od.direction = (
        "LONG"  if od.total_score >= CFG["min_score_long"] else
        "SHORT" if od.total_score <= -CFG["min_score_short"] else
        "FLAT"
    )
    return od


# ─────────────────────────────────────────────
# RISK HESAPLAMA
# ─────────────────────────────────────────────
def compute_risk(
    df: pd.DataFrame,
    direction: str,
    score: float
) -> dict:
    """SL / TP / Leverage hesapla."""
    atr_val = _atr(df).iloc[-1]
    close   = df["close"].iloc[-1]

    sl_dist = max(atr_val * CFG["atr_sl_mult"], close * CFG["sl_min_pct"])

    if direction == "LONG":
        sl  = close - sl_dist
        tp1 = close + sl_dist * CFG["tp1_r"]
        tp2 = close + sl_dist * CFG["tp2_r"]
        tp3 = close + sl_dist * CFG["tp3_r"]
    else:
        sl  = close + sl_dist
        tp1 = close - sl_dist * CFG["tp1_r"]
        tp2 = close - sl_dist * CFG["tp2_r"]
        tp3 = close - sl_dist * CFG["tp3_r"]

    # Skor bazlı kaldıraç
    score_abs = abs(score)
    raw_lev   = CFG["lev_min"] + (score_abs / 12.0) * (CFG["lev_max"] - CFG["lev_min"])
    leverage  = int(np.clip(raw_lev, CFG["lev_min"], CFG["lev_max"]))

    return {
        "entry":    round(close, 4),
        "sl":       round(sl, 4),
        "tp1":      round(tp1, 4),
        "tp2":      round(tp2, 4),
        "tp3":      round(tp3, 4),
        "leverage": leverage,
        "atr":      round(atr_val, 4),
    }


# ─────────────────────────────────────────────
# ANA FONKSİYON
# ─────────────────────────────────────────────
def analyze(
    df: pd.DataFrame,
    symbol: str = "ETHUSDT",
    position: Optional[str] = None,
    entry_price: float = 0.0,
    funding_rate: float = 0.0,
    oi_current: float = 0.0,
    oi_prev: float = 0.0,
) -> SignalResult:
    """
    Tüm analizi çalıştır ve SignalResult döndür.

    Args:
        df:           OHLCV + orderflow kolonları içeren DataFrame
        symbol:       İşlem sembolü
        position:     Açık pozisyon (varsa)
        entry_price:  Giriş fiyatı (dynamic exit için)
    """
    result = SignalResult(symbol=symbol)

    if len(df) < 30:
        return result

    # 1. Orderflow skoru
    od = score_orderflow(df)

    # 2. Absorption
    ab = detect_absorption(df)

    # Absorption bonusunu skora ekle (yön uyumlu olmalı)
    if ab.bullish and od.total_score > 0:
        od.score_absorption = ab.score_bonus
        od.total_score     += ab.score_bonus
    elif ab.bearish and od.total_score < 0:
        od.score_absorption = -ab.score_bonus
        od.total_score     -= ab.score_bonus

    # 3. Funding Rate + OI
    foi = score_funding_oi(
        funding_rate  = funding_rate,
        oi_current    = oi_current,
        oi_prev       = oi_prev,
        price_direction = od.delta,
    )
    od.score_funding = foi.score_funding
    od.score_oi      = foi.score_oi
    od.total_score   = round(od.total_score + foi.score_funding + foi.score_oi, 3)

    # Yeniden direction belirle
    od.direction = (
        "LONG"  if od.total_score >= CFG["min_score_long"] else
        "SHORT" if od.total_score <= -CFG["min_score_short"] else
        "FLAT"
    )

    result.orderflow  = od
    result.absorption = ab
    result.funding_oi = foi
    result.score      = od.total_score
    result.direction  = od.direction

    # 3. Risk parametreleri
    cur_price = df["close"].iloc[-1]
    if od.direction != "FLAT":
        risk = compute_risk(df, od.direction, od.total_score)
        result.entry    = risk["entry"]
        result.sl       = risk["sl"]
        result.tp1      = risk["tp1"]
        result.tp2      = risk["tp2"]
        result.tp3      = risk["tp3"]
        result.leverage = risk["leverage"]
        result.atr      = risk["atr"]
    else:
        result.entry = round(cur_price, 4)  # FLAT'ta da fiyatı göster
        result.atr   = round(_atr(df).iloc[-1], 4)

    # 4. Dynamic exit (açık pozisyon varsa)
    if position in ("LONG", "SHORT") and entry_price > 0:
        dex = compute_dynamic_exit(df, position, entry_price)
        result.dynamic_exit = dex
        if dex.should_exit:
            result.exit_signal = True
            result.exit_reason = dex.reason
    else:
        # Exit verilerini her zaman hesapla (bilgi amaçlı)
        pos_guess = od.direction if od.direction != "FLAT" else "LONG"
        result.dynamic_exit = compute_dynamic_exit(df, pos_guess, df["close"].iloc[-1])

    # Timestamp
    if "timestamp" in df.columns or hasattr(df.index, "name"):
        try:
            result.timestamp = str(df.index[-1])
        except Exception:
            result.timestamp = ""

    return result


# ─────────────────────────────────────────────
# KONSOL ÇIKTISI
# ─────────────────────────────────────────────
def print_signal(res: SignalResult) -> None:
    """Renkli konsol çıktısı."""
    sep = "─" * 56

    dir_color = {
        "LONG":  Fore.GREEN,
        "SHORT": Fore.RED,
        "FLAT":  Fore.YELLOW,
    }.get(res.direction, Fore.WHITE)

    print(f"\n{sep}")
    print(f"{dir_color}{'█'*4} {res.symbol} — {res.direction}  "
          f"(Skor: {res.score:+.2f}){Style.RESET_ALL}")
    print(sep)

    od = res.orderflow
    ab = res.absorption
    dex = res.dynamic_exit

    # ── ORDERFLOW BİLEŞENLERİ ────────────────────────────────────────
    print(f"\n{Fore.CYAN}[ ORDERFLOW ANALİZİ ]{Style.RESET_ALL}")
    rows = [
        ("Delta",          od.delta,         od.score_delta,
         f"delta_pct: %{od.delta_pct*100:.1f}"),
        ("CVD Momentum",   od.cvd_slope,      od.score_cvd,
         f"cvd: {od.cvd:.0f}  slope({CFG['cvd_lookback']}bar): {od.cvd_slope:+.0f}"),
        ("Imbalance",      od.imbalance_ratio, od.score_imbalance,
         f"ratio: {od.imbalance_ratio:.3f}  "
         f"(bull>{CFG['imbalance_bull']} / bear<{CFG['imbalance_bear']})"),
        ("Stacked Imbalance", int(od.stacked_up or od.stacked_dn), od.score_stacked,
         f"up:{od.stacked_up}  dn:{od.stacked_dn}"),
        ("Session Delta",  od.session_delta,  od.score_session,
         f"yön: {od.session_delta_direction}"),
        ("Price vs VWAP",  0, od.score_vwap,
         f"fiyat: {res.entry:.2f}  vwap: {od.vwap:.2f}  → {od.price_vs_vwap}"),
        ("Vol Spike+Delta",od.volume_ratio,   od.score_vol_spike,
         f"vol {od.volume_ratio:.2f}x avg"),
        ("Bar Kapanış",    od.bar_close_pos,  od.score_bar_close,
         f"pos: %{od.bar_close_pos*100:.0f}"),
        ("Delta MA3/MA10", 0, od.score_delta_ma,
         "hızlı/yavaş delta MA"),
    ]

    for name, val, score, detail in rows:
        s_color = Fore.GREEN if score > 0 else (Fore.RED if score < 0 else Fore.WHITE)
        score_str = f"{score:+.2f}" if score != 0 else " 0.00"
        print(f"  {name:<22} {s_color}{score_str}{Style.RESET_ALL}  {detail}")

    # ── ABSORPTION ───────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}[ ABSORPTION ]{Style.RESET_ALL}")
    ab_color = Fore.GREEN if ab.bullish else (Fore.RED if ab.bearish else Fore.WHITE)
    print(f"  {ab_color}{ab.detail}{Style.RESET_ALL}")
    if ab.score_bonus > 0:
        bonus_sign = "+" if (ab.bullish) else "-"
        print(f"  Skor bonusu: {bonus_sign}{ab.score_bonus:.1f}")

    # ── FUNDING RATE + OI ────────────────────────────────────────────
    foi = res.funding_oi
    print(f"\n{Fore.CYAN}[ FUNDING RATE + OI ]{Style.RESET_ALL}")
    if foi.funding_rate == 0.0 and foi.oi_current == 0.0:
        print(f"  Veri yok (signal_logger.py üzerinden çekilir)")
    else:
        fr_color = Fore.GREEN if foi.score_funding > 0 else (Fore.RED if foi.score_funding < 0 else Fore.WHITE)
        oi_color = Fore.GREEN if foi.score_oi > 0 else (Fore.RED if foi.score_oi < 0 else Fore.WHITE)
        print(f"  Funding Rate : {foi.funding_rate_pct:+.4f}%  ({foi.funding_bias})  "
              f"→ {fr_color}skor:{foi.score_funding:+.2f}{Style.RESET_ALL}")
        print(f"  OI Değişimi  : {foi.oi_change_pct:+.3f}%  ({foi.oi_trend})  "
              f"→ {oi_color}skor:{foi.score_oi:+.2f}{Style.RESET_ALL}")

    # ── TOPLAM SKOR ──────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}[ TOPLAM SKOR ]{Style.RESET_ALL}")
    breakdown = (
        f"  Delta:{od.score_delta:+.2f} | CVD:{od.score_cvd:+.2f} | "
        f"Imb:{od.score_imbalance:+.2f} | Stack:{od.score_stacked:+.2f} | "
        f"Sess:{od.score_session:+.2f}\n"
        f"  VWAP:{od.score_vwap:+.2f} | VolSpike:{od.score_vol_spike:+.2f} | "
        f"BarClose:{od.score_bar_close:+.2f} | DeltaMA:{od.score_delta_ma:+.2f} | "
        f"Absorb:{od.score_absorption:+.2f}\n"
        f"  Funding:{od.score_funding:+.2f} | OI:{od.score_oi:+.2f}"
    )
    print(breakdown)
    print(f"  {dir_color}TOPLAM: {res.score:+.2f}  →  {res.direction}{Style.RESET_ALL}")

    # ── RISK ─────────────────────────────────────────────────────────
    if res.direction != "FLAT":
        print(f"\n{Fore.CYAN}[ RİSK ]{Style.RESET_ALL}")
        print(f"  Giriş:  {res.entry:.4f}")
        print(f"  SL:     {res.sl:.4f}  (ATR x{CFG['atr_sl_mult']})")
        print(f"  TP1:    {res.tp1:.4f}  ({CFG['tp1_r']}R)")
        print(f"  TP2:    {res.tp2:.4f}  ({CFG['tp2_r']}R)")
        print(f"  TP3:    {res.tp3:.4f}  ({CFG['tp3_r']}R)")
        print(f"  Kaldıraç: {res.leverage}x")

    # ── DYNAMIC EXIT ─────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}[ DYNAMIC EXIT ]{Style.RESET_ALL}")
    ex_color = Fore.RED if dex.should_exit else Fore.WHITE
    print(f"  CVD Divergence : {'✅ ' + dex.cvd_div_type if dex.cvd_divergence else '—'}")
    print(f"  ATR Trail LONG : {dex.atr_trail_long:.4f}  "
          f"(ATR {dex.atr_value:.4f} x {CFG['atr_trail_mult']})")
    print(f"  ATR Trail SHORT: {dex.atr_trail_short:.4f}")
    print(f"  BPR/IFVG Zone  : "
          f"{'✅ ' + str(dex.bpr_zone_bot) + '-' + str(dex.bpr_zone_top) if dex.bpr_zone_hit else '—'}")
    if dex.should_exit:
        print(f"  {ex_color}⚠️  EXIT: {dex.reason}{Style.RESET_ALL}")
    else:
        print(f"  Durum: Bekleme")

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────
# TELEGRAM MESAJ FORMATTERI
# ─────────────────────────────────────────────
def format_telegram_signal(res: SignalResult) -> str:
    """Telegram için signal mesajı üret."""
    od  = res.orderflow
    ab  = res.absorption
    dex = res.dynamic_exit

    dir_emoji = {"LONG": "🟢", "SHORT": "🔴", "FLAT": "🟡"}.get(res.direction, "⚪")
    ab_str = ""
    if ab.bullish:
        ab_str = f"\n📦 *Absorption: Bullish* ✅ (Vol {ab.volume_ratio:.2f}x)"
    elif ab.bearish:
        ab_str = f"\n📦 *Absorption: Bearish* ✅ (Vol {ab.volume_ratio:.2f}x)"

    score_breakdown = (
        f"Delta:{od.score_delta:+.1f} CVD:{od.score_cvd:+.1f} "
        f"Imb:{od.score_imbalance:+.1f} Stack:{od.score_stacked:+.1f} "
        f"Sess:{od.score_session:+.1f}\n"
        f"VWAP:{od.score_vwap:+.1f} VolSpk:{od.score_vol_spike:+.1f} "
        f"BarClose:{od.score_bar_close:+.1f} ΔMA:{od.score_delta_ma:+.1f} "
        f"Absorb:{od.score_absorption:+.1f}\n"
        f"Funding:{od.score_funding:+.1f} OI:{od.score_oi:+.1f}"
    )

    foi = res.funding_oi
    foi_str = ""
    if foi.funding_rate != 0.0 or foi.oi_current != 0.0:
        foi_str = (
            f"\n📈 *Funding:* `{foi.funding_rate_pct:+.4f}%` ({foi.funding_bias})"
            f"\n📊 *OI:* `{foi.oi_change_pct:+.3f}%` ({foi.oi_trend})"
        )

    risk_str = ""
    if res.direction != "FLAT":
        risk_str = (
            f"\n\n💰 *Giriş:* `{res.entry}`\n"
            f"🛑 *SL:* `{res.sl}`\n"
            f"🎯 *TP1:* `{res.tp1}` | *TP2:* `{res.tp2}` | *TP3:* `{res.tp3}`\n"
            f"⚡ *Kaldıraç:* {res.leverage}x"
        )

    exit_str = ""
    if dex.cvd_divergence or dex.bpr_zone_hit:
        exit_str = f"\n\n📊 *Exit İzleme:*"
        if dex.cvd_divergence:
            exit_str += f"\n• CVD Div: {dex.cvd_div_type}"
        exit_str += f"\n• ATR Trail: {dex.atr_trail_long:.2f} / {dex.atr_trail_short:.2f}"
        if dex.bpr_zone_hit:
            exit_str += f"\n• BPR/IFVG Zone: {dex.bpr_zone_bot:.2f}–{dex.bpr_zone_top:.2f}"

    msg = (
        f"{dir_emoji} *{res.symbol} — {res.direction}*\n"
        f"Skor: `{res.score:+.2f}` | {res.timestamp}"
        f"{ab_str}\n\n"
        f"```\n{score_breakdown}\n```"
        f"{foi_str}"
        f"{risk_str}"
        f"{exit_str}"
    )
    return msg


def format_telegram_exit(res: SignalResult) -> str:
    """Telegram için exit sinyali mesajı üret."""
    if not res.exit_signal:
        return ""

    dex = res.dynamic_exit
    msg = (
        f"⚠️ *EXIT SİNYALİ — {res.symbol}*\n"
        f"Sebep: `{res.exit_reason}`\n\n"
        f"ATR: `{dex.atr_value:.4f}`\n"
        f"Trail LONG: `{dex.atr_trail_long:.4f}`\n"
        f"Trail SHORT: `{dex.atr_trail_short:.4f}`"
    )
    if dex.bpr_zone_hit:
        msg += f"\nBPR/IFVG: `{dex.bpr_zone_bot:.4f}–{dex.bpr_zone_top:.4f}`"
    return msg


# ─────────────────────────────────────────────
# MULTI-TIMEFRAME SİSTEM
# ─────────────────────────────────────────────
@dataclass
class MTFResult:
    symbol:      str = "ETHUSDT"
    timestamp:   str = ""
    res_1d:      SignalResult = field(default_factory=SignalResult)
    res_4h:      SignalResult = field(default_factory=SignalResult)
    res_1h:      SignalResult = field(default_factory=SignalResult)
    res_15m:     SignalResult = field(default_factory=SignalResult)
    res_5m:      SignalResult = field(default_factory=SignalResult)
    confluence:  int = 0          # kaç timeframe aynı yönde (0-5)
    direction:   str = "FLAT"
    should_send: bool = False
    note:        str = ""         # "GÜÇLÜ" | "ORTA" | "ZAYIF"


def analyze_mtf(
    df_1h:  pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m:  pd.DataFrame,
    symbol: str = "ETHUSDT",
    funding_rate: float = 0.0,
    oi_current:   float = 0.0,
    oi_prev:      float = 0.0,
    position:     Optional[str] = None,
    entry_price:  float = 0.0,
    df_4h:  pd.DataFrame = None,
    df_1d:  pd.DataFrame = None,
) -> MTFResult:
    """
    5 timeframe analiz + confluence hesapla.

    1D  → Makro trend
    4H  → Orta vade yön
    1H  → Kısa vade trend
    15M → Setup tetikleyici
    5M  → Giriş zamanlaması

    Confluence:
      5/5 → GÜÇLÜ 🔥
      4/5 → GÜÇLÜ 🔥
      3/5 → ORTA  ⚡ → Telegram'a gönder
      2/5 veya altı → Gönderme
    """
    mtf = MTFResult(symbol=symbol)

    # FR + OI sadece 5M analizine eklenir
    mtf.res_5m  = analyze(
        df_5m, symbol=symbol,
        funding_rate=funding_rate,
        oi_current=oi_current,
        oi_prev=oi_prev,
        position=position,
        entry_price=entry_price,
    )
    mtf.res_15m = analyze(df_15m, symbol=symbol)
    mtf.res_1h  = analyze(df_1h,  symbol=symbol)
    mtf.res_4h  = analyze(df_4h,  symbol=symbol) if df_4h is not None and len(df_4h) >= 30 else SignalResult(symbol=symbol)
    mtf.res_1d  = analyze(df_1d,  symbol=symbol) if df_1d is not None and len(df_1d) >= 30 else SignalResult(symbol=symbol)

    # Timestamp
    try:
        mtf.timestamp = str(df_5m.index[-1])
    except Exception:
        pass

    # Confluence hesapla — aktif TF'ler
    active_results = [mtf.res_5m, mtf.res_15m, mtf.res_1h]
    if df_4h is not None and len(df_4h) >= 30:
        active_results.append(mtf.res_4h)
    if df_1d is not None and len(df_1d) >= 30:
        active_results.append(mtf.res_1d)

    total_tf    = len(active_results)
    dirs        = [r.direction for r in active_results]
    long_count  = dirs.count("LONG")
    short_count = dirs.count("SHORT")

    if long_count >= short_count:
        mtf.direction  = "LONG" if long_count > 0 else "FLAT"
        mtf.confluence = long_count
    else:
        mtf.direction  = "SHORT"
        mtf.confluence = short_count

    # Gönderim eşiği: 3/5 (veya 3/total_tf)
    min_conf = 3
    if mtf.confluence >= total_tf:
        mtf.should_send = True
        mtf.note        = "GÜÇLÜ"
    elif mtf.confluence == total_tf - 1:
        mtf.should_send = True
        mtf.note        = "GÜÇLÜ"
    elif mtf.confluence >= min_conf:
        mtf.should_send = True
        mtf.note        = "ORTA"
    else:
        mtf.should_send = False
        mtf.note        = "ZAYIF"

    return mtf


def print_mtf(mtf: MTFResult) -> None:
    """MTF sonuçlarını konsola yazdır."""
    sep = "═" * 56
    dir_color = {
        "LONG":  Fore.GREEN,
        "SHORT": Fore.RED,
        "FLAT":  Fore.YELLOW,
    }.get(mtf.direction, Fore.WHITE)

    print(f"\n{sep}")
    print(f"{dir_color}{'█'*4} {mtf.symbol} — MTF ANALİZİ{Style.RESET_ALL}")
    print(sep)

    tf_labels = [
        ("1 Gün   ", "Makro Trend",    mtf.res_1d),
        ("4 Saat  ", "Orta Vade",      mtf.res_4h),
        ("1 Saat  ", "Trend Bias",     mtf.res_1h),
        ("15 Dakika", "Setup",         mtf.res_15m),
        ("5 Dakika", "Giriş Zamanı",   mtf.res_5m),
    ]

    for tf_label, role, res in tf_labels:
        if res.direction == "FLAT" and res.score == 0.0:
            print(f"  ⏰ {tf_label:<10} ⚪ {'—':<6}  [{role}] (veri yok)")
            continue
        dc    = {"LONG": Fore.GREEN, "SHORT": Fore.RED, "FLAT": Fore.YELLOW}.get(res.direction, Fore.WHITE)
        emoji = {"LONG": "🟢", "SHORT": "🔴", "FLAT": "🟡"}.get(res.direction, "⚪")
        ab_tag = " 📦ABSORB" if (res.absorption.bullish or res.absorption.bearish) else ""
        print(f"  ⏰ {tf_label:<10} {emoji} {dc}{res.direction:<6}{Style.RESET_ALL} "
              f"(skor:{res.score:+.2f})  [{role}]{ab_tag}")

    total_tf = sum(1 for r in [mtf.res_1d, mtf.res_4h, mtf.res_1h, mtf.res_15m, mtf.res_5m]
                   if not (r.direction == "FLAT" and r.score == 0.0))
    print(f"\n  🔗 Confluence : {mtf.confluence}/{total_tf} {mtf.direction}")
    conf_color = Fore.GREEN if mtf.confluence >= total_tf - 1 else (Fore.YELLOW if mtf.confluence >= 3 else Fore.RED)
    print(f"  📶 Güç        : {conf_color}{mtf.note}{Style.RESET_ALL}")
    print(f"  📤 Gönderilsin: {'✅' if mtf.should_send else '❌'}")

    dominant = mtf.res_5m if mtf.res_5m.direction != "FLAT" else (
               mtf.res_15m if mtf.res_15m.direction != "FLAT" else mtf.res_1h)
    if dominant.direction != "FLAT":
        print(f"\n  💰 Giriş: {dominant.entry:.4f}  SL: {dominant.sl:.4f}  "
              f"TP1: {dominant.tp1:.4f}  Kaldıraç: {dominant.leverage}x")

    print(f"\n{sep}\n")


def format_telegram_mtf(mtf: MTFResult) -> str:
    """MTF için Telegram mesajı üret."""
    if not mtf.should_send:
        return ""

    dir_emoji  = {"LONG": "🟢", "SHORT": "🔴", "FLAT": "🟡"}.get(mtf.direction, "⚪")
    note_emoji = {"GÜÇLÜ": "🔥", "ORTA": "⚡", "ZAYIF": "💤"}.get(mtf.note, "")

    def tf_line(res):
        if res.direction == "FLAT" and res.score == 0.0:
            return "⚪ `—    ` veri yok"
        e  = {"LONG": "🟢", "SHORT": "🔴", "FLAT": "🟡"}.get(res.direction, "⚪")
        ab = " 📦" if (res.absorption.bullish or res.absorption.bearish) else ""
        return f"{e} `{res.direction:<5}` skor:`{res.score:+.2f}`{ab}"

    total_tf = sum(1 for r in [mtf.res_1d, mtf.res_4h, mtf.res_1h, mtf.res_15m, mtf.res_5m]
                   if not (r.direction == "FLAT" and r.score == 0.0))

    # Risk
    dominant = mtf.res_5m if mtf.res_5m.direction != "FLAT" else (
               mtf.res_15m if mtf.res_15m.direction != "FLAT" else mtf.res_1h)
    risk_str = ""
    if dominant.direction != "FLAT":
        risk_str = (
            f"\n\n💰 *Giriş:* `{dominant.entry}`\n"
            f"🛑 *SL:* `{dominant.sl}`\n"
            f"🎯 *TP1:* `{dominant.tp1}` | *TP2:* `{dominant.tp2}` | *TP3:* `{dominant.tp3}`\n"
            f"⚡ *Kaldıraç:* {dominant.leverage}x"
        )

    # Funding + OI
    foi = mtf.res_5m.funding_oi
    foi_str = ""
    if foi.funding_rate != 0.0 or foi.oi_current != 0.0:
        foi_str = (
            f"\n📈 *Funding:* `{foi.funding_rate_pct:+.4f}%` ({foi.funding_bias})"
            f"  📊 *OI:* `{foi.oi_change_pct:+.3f}%` ({foi.oi_trend})"
        )

    # Exit izleme
    dex = mtf.res_5m.dynamic_exit
    exit_str = ""
    if dex.cvd_divergence or dex.bpr_zone_hit:
        exit_str = f"\n\n🚨 *Exit İzleme:*"
        if dex.cvd_divergence:
            exit_str += f"\n• CVD Div: `{dex.cvd_div_type}`"
        exit_str += f"\n• ATR Trail: `{dex.atr_trail_long:.2f}` / `{dex.atr_trail_short:.2f}`"
        if dex.bpr_zone_hit:
            exit_str += f"\n• BPR/IFVG: `{dex.bpr_zone_bot:.2f}–{dex.bpr_zone_top:.2f}`"

    msg = (
        f"{dir_emoji} *{mtf.symbol} — Multi-Timeframe* {note_emoji}\n"
        f"{mtf.timestamp}\n\n"
        f"📅 *1 Gün    :* {tf_line(mtf.res_1d)}\n"
        f"⏰ *4 Saat   :* {tf_line(mtf.res_4h)}\n"
        f"⏰ *1 Saat   :* {tf_line(mtf.res_1h)}\n"
        f"⏰ *15 Dakika:* {tf_line(mtf.res_15m)}\n"
        f"⏰ *5 Dakika :* {tf_line(mtf.res_5m)}\n\n"
        f"🔗 *Confluence:* {mtf.confluence}/{total_tf} {mtf.direction}\n"
        f"💡 Karar senindir"
        f"{foi_str}"
        f"{risk_str}"
        f"{exit_str}"
    )
    return msg


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import requests as _requests

    def _fetch_live(symbol="ETHUSDT", interval="5", limit=300):
        url = "https://api.bybit.com/v5/market/kline"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = _requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        rows = r.json()["result"]["list"]
        df = pd.DataFrame(rows, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df = df.iloc[::-1].reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
        df = df.set_index("timestamp")
        df["delta"] = df.apply(
            lambda r: r["volume"] * 0.6 if r["close"] >= r["open"] else -r["volume"] * 0.6, axis=1)
        df["buy_volume"]  = df["volume"] * df["delta"].apply(lambda d: 0.7 if d > 0 else 0.3)
        df["sell_volume"] = df["volume"] - df["buy_volume"]
        df["imbalance_ratio"] = df["buy_volume"] / df["volume"]
        df["cvd"] = df["delta"].cumsum()
        df["session_delta"] = df["delta"].rolling(12).sum()
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        bull_streak = (df["imbalance_ratio"] > 0.58).astype(int)
        bear_streak = (df["imbalance_ratio"] < 0.42).astype(int)
        df["stacked_imbalance_up"] = bull_streak.rolling(3).sum() == 3
        df["stacked_imbalance_dn"] = bear_streak.rolling(3).sum() == 3
        return df

    print("Bybit'ten canlı veri çekiliyor...")
    df_live = _fetch_live()
    print(f"  {len(df_live)} bar | son fiyat: {df_live['close'].iloc[-1]:.2f}")

    result = analyze(df_live, symbol="ETHUSDT")
    print_signal(result)

    print("─── TELEGRAM SİNYAL MESAJI ───")
    print(format_telegram_signal(result))

    if result.exit_signal:
        print("\n─── TELEGRAM EXIT MESAJI ───")
        print(format_telegram_exit(result))