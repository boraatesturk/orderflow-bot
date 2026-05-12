@echo off
if "%1"=="egit"   python ml_model.py --train
if "%1"=="tahmin" python ml_model.py --predict
if "%1"=="sinyal" python signal_engine.py --once
if "%1"=="canli"  python signal_engine.py --live
if "%1"=="liq"    python sr_liquidity.py
if "%1"=="veri"   python data_collector.py