# ORDERFLOW BOT — KURULUM & KULLANIM REHBERI
## ETHUSDT | Binance | Python | Visual Studio

---

## KLASOR YAPISI

```
orderflow_bot/
├── data_collector.py    ← 1. ONCE BU
├── signal_engine.py     ← 2. SONRA BU
├── requirements.txt     ← Kutuphaneler
└── data/                ← Otomatik olusur
    ├── ETHUSDT_orderflow_180d.parquet
    └── ETHUSDT_orderflow_180d.csv
```

---

## 1. ADIM: Python Ortami Kur (Visual Studio'da)

Visual Studio'da **Tools → Python → Python Environments** ac.
Ya da terminal ac ve:

```bash
# Proje klasorunde:
python -m venv .venv

# Sanal ortami aktive et:
# Windows:
.venv\Scripts\activate

# Kutuphaneleri yukle:
pip install -r requirements.txt
```

---

## 2. ADIM: Veriyi Cek

```bash
python data_collector.py
```

**Ne yapar:**
- Binance'den 180 gunluk 1dk ETHUSDT mum verisi ceker
- Her mum icin orderflow feature'larini hesaplar
- `data/` klasorune `.parquet` ve `.csv` kaydeder

**Suresi:** ~5-15 dakika (internet hiziniza gore)

**aggTrades notu:**  
`aggTrades` cekilmesi 10-30 dakika surebilir. Ctrl+C ile durdurup
sadece OHLCV bazli tahmine gececek sekilde tasarlanmistir.
Full aggTrades = daha hassas delta hesabi.

---

## 3. ADIM: Sinyal Uret

**Canli mod (son bar):**
```bash
python signal_engine.py
```

**Son 500 bar backtest:**
```bash
python signal_engine.py --last 500
```

**Tum dataset backtest:**
```bash
python signal_engine.py --backtest
```

---

## DATASET KOLONLARI

| Kolon | Aciklama |
|-------|----------|
| open/high/low/close | OHLC fiyat |
| volume | Bar toplam hacim |
| taker_buy_volume | Taker buy hacmi |
| buy_volume | Hesaplanan buy hacmi |
| sell_volume | Hesaplanan sell hacmi |
| delta | buy_vol - sell_vol |
| min_delta | Bar icindeki min kumulatif delta |
| max_delta | Bar icindeki max kumulatif delta |
| session_delta | Gunluk kumulatif delta |
| session_volume | Gunluk kumulatif hacim |
| volume_per_second | Hacim / 60 |
| bid_trades | Satici agresif trade sayisi |
| ask_trades | Alici agresif trade sayisi |
| imbalance_ratio | buy_vol / total_vol |
| stacked_imbalance_up | 3+ bar ust uste bull imbalance |
| stacked_imbalance_dn | 3+ bar ust uste bear imbalance |
| cvd | Cumulative Volume Delta |
| vwap | Session VWAP |
| poc_price | Point of Control fiyati |
| typical_price | (H+L+C)/3 |

---

## SINYAL KURALLARI

```
BUY  skoru >= 6/10  AND  BUY > SELL  →  BUY
SELL skoru >= 6/10  AND  SELL > BUY  →  SELL
Diger                                →  FLAT
```

**Puan veren kurallar:**
1. Delta yonu (pozitif/negatif) — agirlik: 0-1.5
2. CVD momentum (yukseliyor/dusuyor) — agirlik: 1.0
3. Imbalance ratio — agirlik: 1.5
4. Stacked imbalance (3+ bar) — agirlik: 2.0 ⭐
5. Session delta yonu — agirlik: 0.75
6. Fiyat vs VWAP — agirlik: 0.5
7. Volume spike + delta yonu — agirlik: 1.0
8. Bar kapanis pozisyonu — agirlik: 0.5
9. Delta MA crossover — agirlik: 0.5

---

## PARAMETRELERI DEGISTIR

`signal_engine.py` icindeki `CFG` sozlugunu duzenle:

```python
CFG = {
    "min_score_buy":    6,    # 5'e dusurmek = daha fazla sinyal
    "imbalance_bull":   0.55, # 0.60'a cikmak = daha strict
    "stacked_confirm":  True, # False = stacked imbalance kullanma
    ...
}
```

---

## SONRAKI ADIMLAR

- [ ] WebSocket ile realtime stream ekle
- [ ] ML model (sklearn RandomForest) entegre et  
- [ ] Telegram alert entegrasyonu
- [ ] Footprint chart gorsellestirmesi
- [ ] Risk management / stop loss modulu
