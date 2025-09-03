#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
backtest.py — односценарный бэктестер простых правил-ориентированных стратегий.
Источник данных: Yahoo Finance (через yfinance).
Поддержка: одиночные и портфельные тикеры, комиссии/слиппедж, ATR-стоп/тейк, отчёты и графики.

Примеры:
    python backtest.py --ticker AAPL --start 2018-01-01 --strategy sma_cross --fast 10 --slow 30
    python backtest.py --ticker EURUSD=X --start 2022-01-01 --interval 1h --strategy rsi_revert
    python backtest.py --ticker AAPL,MSFT --start 2019-01-01 --weights AAPL=0.6,MSFT=0.4 --stop_atr_k 3
"""

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit(
        "Не удалось импортировать 'yfinance'. Установите пакет: pip install yfinance"
    ) from e


# ============================== УТИЛИТЫ / СТРУКТУРЫ ==============================

@dataclass
class Trade:
    ticker: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: int
    fees_entry: float
    fees_exit: float

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty - (self.fees_entry + self.fees_exit)

    @property
    def ret(self) -> float:
        # доходность сделки относительно вложенной суммы (без плеча)
        denom = self.entry_price * self.qty
        return (self.pnl / denom) if denom > 0 else 0.0


def parse_weights(weights_str: Optional[str], tickers: List[str], logger: logging.Logger) -> Dict[str, float]:
    """Парсинг весов формата 'AAA=0.6,BBB=0.4'. Если не заданы — равные веса."""
    if not weights_str:
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}

    parts = [p.strip() for p in weights_str.split(",") if p.strip()]
    weights: Dict[str, float] = {}
    for p in parts:
        if "=" not in p:
            logger.warning("Вес '%s' пропущен — ожидается формат TICKER=WEIGHT", p)
            continue
        t, v = p.split("=", 1)
        t = t.strip()
        try:
            weights[t] = float(v)
        except ValueError:
            logger.warning("Неверное значение веса для '%s': '%s'", t, v)

    # добавить отсутствующие тикеры с нулевым весом
    for t in tickers:
        weights.setdefault(t, 0.0)

    s = sum(weights.values())
    if s <= 0:
        logger.warning("Сумма весов <= 0. Будут использованы равные веса.")
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}

    if abs(s - 1.0) > 1e-9:
        logger.warning("Сумма весов = %.6f, будет нормализована до 1.0", s)
        for t in list(weights.keys()):
            weights[t] /= s
    return weights


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def max_drawdown(equity: pd.Series) -> float:
    """Максимальная просадка по кривой капитала (от 0 до отрицательного числа)."""
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    return float(drawdown.min() if len(drawdown) else 0.0)


def consecutive_counts(bools: List[bool]) -> Tuple[int, int]:
    """Максимальные серии выигрышей/проигрышей."""
    max_wins = max_losses = cur_wins = cur_losses = 0
    for b in bools:
        if b:
            cur_wins += 1
            cur_losses = 0
        else:
            cur_losses += 1
            cur_wins = 0
        max_wins = max(max_wins, cur_wins)
        max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses


# ============================== ИНДИКАТОРЫ ==============================

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    # Классический RSI по Wilder
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # нейтральное значение при нехватке истории


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    # True Range
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=n, min_periods=n).mean()


# ============================== ГЕНЕРАЦИЯ СИГНАЛОВ ==============================

def signals_sma_cross(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """Сигнал-режим: 1 = хотим быть в лонге, 0 = вне рынка."""
    s_fast = sma(df["Close"], fast)
    s_slow = sma(df["Close"], slow)
    # хотим быть в позиции, когда fast > slow
    sig = (s_fast > s_slow).astype(int)
    sig = sig.fillna(0)
    return sig


def signals_rsi_revert(df: pd.DataFrame, period: int, buy_lvl: float, exit_lvl: float) -> pd.Series:
    r = rsi(df["Close"], period)
    # Состояние: если ниже buy_lvl — хотим войти (1), выходим когда rsi > exit_lvl
    state = []
    in_pos = False
    for val in r:
        if not in_pos:
            if val < buy_lvl:
                in_pos = True
        else:
            if val > exit_lvl:
                in_pos = False
        state.append(1 if in_pos else 0)
    return pd.Series(state, index=df.index).astype(int)


# ============================== ЗАГРУЗКА ДАННЫХ ==============================

def load_data(tickers: List[str], start: Optional[str], end: Optional[str], interval: str,
              logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """Загрузка исторических данных по каждому тикеру — отдельным DataFrame."""
    data: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        logger.info("Загрузка данных для %s ...", t)
        df = yf.download(t, start=start, end=end, interval=interval, progress=False, auto_adjust=False, threads=True)
        if df is None or df.empty:
            logger.error("Пустые данные по тикеру %s. Пропуск.", t)
            continue
        # Приведение типов/индекса
        df = df.rename(columns={c: c.strip().title() for c in df.columns})
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index, utc=False)  # локальная TZ не навязывается
        df = df.sort_index()
        data[t] = df
    if not data:
        raise SystemExit("Не удалось получить данные ни по одному тикеру.")
    return data


# ============================== СИМУЛЯТОР ПОРТФЕЛЯ ==============================

@dataclass
class Position:
    qty: int
    entry_price: float
    entry_time: pd.Timestamp
    fees_entry: float
    stop_level: Optional[float] = None
    take_level: Optional[float] = None


def simulate_portfolio(
    data: Dict[str, pd.DataFrame],
    signals: Dict[str, pd.Series],
    weights: Dict[str, float],
    initial_capital: float,
    fee_perc: float,
    slippage_perc: float,
    position_mode: str,
    risk_pct: float,
    risk_budget: float,
    atr_period: int,
    atr_k: float,
    stop_atr_k: Optional[float],
    take_profit_perc: Optional[float],
    allow_leverage: bool,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, List[Trade], pd.Series]:
    """
    Основная пошаговая симуляция.
    Возвращает:
        - equity_df: столбцы ['equity', 'cash'] + по желанию позиционные столбцы
        - trades: список сделок
        - invested: булева серия (в позиции/вне), для расчёта Exposure
    """
    tickers = list(data.keys())

    # Индексы по тикерам и общий таймлайн (объединение)
    all_times = sorted(set().union(*[df.index for df in data.values()]))
    timeline = pd.DatetimeIndex(all_times)

    # Подготовка ATR по тикерам (для стопов и позиционирования)
    atr_map: Dict[str, pd.Series] = {}
    for t in tickers:
        atr_map[t] = atr(data[t], n=atr_period)

    # Перенос Close на общий таймлайн с ffill для оценки equity
    close_ffill: Dict[str, pd.Series] = {}
    for t in tickers:
        close_ffill[t] = data[t]["Close"].reindex(timeline).ffill()

    # Быстрый доступ к "следующей дате" внутри ИХ индексов (для T+1 исполнения)
    index_map: Dict[str, pd.DatetimeIndex] = {t: data[t].index for t in tickers}

    def next_bar_time(ticker: str, current_time: pd.Timestamp) -> Optional[pd.Timestamp]:
        idx = index_map[ticker]
        # позиция, куда бы встал current_time
        pos = idx.searchsorted(current_time, side="right")
        if pos < len(idx):
            return idx[pos]
        return None

    # Портфельное состояние
    cash = initial_capital
    positions: Dict[str, Optional[Position]] = {t: None for t in tickers}
    trades: List[Trade] = []
    invested_flags: List[bool] = []

    # Очередь отложенных ордеров: {time: [(ticker, action)]}
    pending: Dict[pd.Timestamp, List[Tuple[str, str]]] = {}

    # Основной цикл по времени
    equity_records: List[Tuple[pd.Timestamp, float, float]] = []

    for t in timeline:
        # 1) Исполнение отложенных ордеров на этом баре (по цене Open[t])
        for (ticker, action) in list(pending.get(t, [])):
            df_t = data[ticker]
            if t not in df_t.index:
                # защиты от рассинхронизации (не должно происходить, но на всякий случай)
                logger.debug("Нет бара %s для %s при исполнении отложенного ордера.", t, ticker)
                continue

            o = float(df_t.loc[t, "Open"])
            exec_price = o * (1 + slippage_perc if action == "buy" else (1 - slippage_perc))
            cur_pos = positions[ticker]
            w = weights.get(ticker, 0.0)

            if action == "buy" and cur_pos is None:
                # Определяем размер позиции
                if position_mode == "fixed":
                    # бюджет на покупку для данного тикера
                    budget = cash * (risk_pct / 100.0) * w
                    qty = int(math.floor(budget / exec_price))
                elif position_mode == "atr":
                    # риск-бюджет на тикер, делённый на ATR
                    atr_val = float(atr_map[ticker].get(t, np.nan))
                    if not np.isfinite(atr_val) or atr_val <= 0:
                        logger.debug("Нет ATR для %s на %s — покупка пропущена.", ticker, t)
                        qty = 0
                    else:
                        qty = int(math.floor((risk_budget * w) / (atr_k * atr_val)))
                else:
                    qty = 0

                if qty <= 0:
                    logger.debug("Нулевая позиция по %s на %s — пропуск входа.", ticker, t)
                else:
                    # Проверка на кэш/плечо
                    fees_entry = exec_price * qty * fee_perc
                    cost = exec_price * qty + fees_entry
                    if not allow_leverage and cost > cash:
                        # уменьшить qty до доступного кэша
                        qty_afford = int(math.floor(cash / (exec_price * (1 + fee_perc))))
                        qty = max(0, min(qty, qty_afford))
                        fees_entry = exec_price * qty * fee_perc
                        cost = exec_price * qty + fees_entry

                    if qty > 0 and (allow_leverage or cost <= cash):
                        cash -= cost
                        # Стоп/тейк уровни
                        stop_level = None
                        take_level = None
                        if stop_atr_k is not None:
                            atr_val = float(atr_map[ticker].get(t, np.nan))
                            if np.isfinite(atr_val):
                                stop_level = exec_price - stop_atr_k * atr_val
                        if take_profit_perc is not None:
                            take_level = exec_price * (1 + take_profit_perc)

                        positions[ticker] = Position(
                            qty=qty, entry_price=exec_price, entry_time=t, fees_entry=fees_entry,
                            stop_level=stop_level, take_level=take_level
                        )
                        logger.debug("BUY %s @ %.4f x %d (fees=%.4f)", ticker, exec_price, qty, fees_entry)
                    else:
                        logger.debug("Недостаточно кэша для входа %s на %s.", ticker, t)

            elif action == "sell" and cur_pos is not None:
                qty = cur_pos.qty
                fees_exit = exec_price * qty * fee_perc
                cash += exec_price * qty - fees_exit
                trade = Trade(
                    ticker=ticker,
                    entry_time=cur_pos.entry_time,
                    exit_time=t,
                    entry_price=cur_pos.entry_price,
                    exit_price=exec_price,
                    qty=qty,
                    fees_entry=cur_pos.fees_entry,
                    fees_exit=fees_exit,
                )
                trades.append(trade)
                positions[ticker] = None
                logger.debug("SELL %s @ %.4f x %d (fees=%.4f) | PnL=%.4f",
                             ticker, exec_price, qty, fees_exit, trade.pnl)

        # Очищаем обработанные отложенные ордера на этот момент
        if t in pending:
            del pending[t]

        # 2) Проверка стоп/тейк по открытым позициям на текущем баре
        for ticker in tickers:
            cur_pos = positions.get(ticker)
            if cur_pos is None:
                continue
            df_t = data[ticker]
            if t not in df_t.index:
                # нет бара — нечего проверять
                continue
            hi = float(df_t.loc[t, "High"])
            lo = float(df_t.loc[t, "Low"])

            exit_reason = None
            exit_price = None

            # Стоп-лосс имеет приоритет
            if cur_pos.stop_level is not None and lo <= cur_pos.stop_level:
                exit_reason = "stop"
                exit_price = float(cur_pos.stop_level)
            elif cur_pos.take_level is not None and hi >= cur_pos.take_level:
                exit_reason = "take"
                exit_price = float(cur_pos.take_level)

            if exit_reason is not None:
                qty = cur_pos.qty
                fees_exit = exit_price * qty * fee_perc
                cash += exit_price * qty - fees_exit
                trade = Trade(
                    ticker=ticker,
                    entry_time=cur_pos.entry_time,
                    exit_time=t,
                    entry_price=cur_pos.entry_price,
                    exit_price=exit_price,
                    qty=qty,
                    fees_entry=cur_pos.fees_entry,
                    fees_exit=fees_exit,
                )
                trades.append(trade)
                positions[ticker] = None
                logger.debug("EXIT(%s) %s @ %.4f x %d (fees=%.4f) | PnL=%.4f",
                             exit_reason.upper(), ticker, exit_price, qty, fees_exit, trade.pnl)

        # 3) Создание новых отложенных ордеров на основе желаемого сигнала (режима)
        for ticker in tickers:
            desired = int(signals[ticker].get(t, 0))
            cur_pos = positions[ticker]
            # уже есть отложенный ордер на ближайший бар? не дублируем
            already_pending = False
            for ts, lst in pending.items():
                if any(p_t == ticker for p_t, _ in lst):
                    already_pending = True
                    break

            if desired == 1 and cur_pos is None and not already_pending:
                nt = next_bar_time(ticker, t)
                if nt is not None:
                    pending.setdefault(nt, []).append((ticker, "buy"))
            elif desired == 0 and cur_pos is not None and not already_pending:
                nt = next_bar_time(ticker, t)
                if nt is not None:
                    pending.setdefault(nt, []).append((ticker, "sell"))

        # 4) Оценка equity и фиксация признака "в позиции"
        total_position_value = 0.0
        any_invested = False
        for ticker in tickers:
            cur_pos = positions.get(ticker)
            px = float(close_ffill[ticker].get(t, np.nan))
            if cur_pos is not None and np.isfinite(px):
                total_position_value += cur_pos.qty * px
                any_invested = True
        equity = cash + total_position_value
        equity_records.append((t, equity, cash))
        invested_flags.append(any_invested)

    # 5) Закрыть все позиции на последнем доступном Close (с учётом слиппеджа в сторону продажи)
    if any(positions[t] is not None for t in tickers):
        last_t = timeline[-1]
        for ticker in tickers:
            cur_pos = positions.get(ticker)
            if cur_pos is None:
                continue
            px_last = float(close_ffill[ticker].get(last_t, np.nan))
            if np.isfinite(px_last) and cur_pos.qty > 0:
                exec_price = px_last * (1 - slippage_perc)  # как будто продаём по рынку в конце
                fees_exit = exec_price * cur_pos.qty * fee_perc
                cash += exec_price * cur_pos.qty - fees_exit
                trade = Trade(
                    ticker=ticker,
                    entry_time=cur_pos.entry_time,
                    exit_time=last_t,
                    entry_price=cur_pos.entry_price,
                    exit_price=exec_price,
                    qty=cur_pos.qty,
                    fees_entry=cur_pos.fees_entry,
                    fees_exit=fees_exit,
                )
                trades.append(trade)
                positions[ticker] = None
                logging.debug("FORCE EXIT %s @ %.4f x %d | PnL=%.4f",
                              ticker, exec_price, cur_pos.qty, trade.pnl)
        # пересчитать финальную точку equity
        total_position_value = 0.0
        equity = cash + total_position_value
        equity_records[-1] = (equity_records[-1][0], equity, cash)

    equity_df = pd.DataFrame(equity_records, columns=["time", "equity", "cash"]).set_index("time")
    invested = pd.Series(invested_flags, index=timeline)

    return equity_df, trades, invested


# ============================== РАСЧЁТ МЕТРИК ==============================

def compute_metrics(
    equity_df: pd.DataFrame,
    trades: List[Trade],
    rf: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, float]:
    """Расчёт основных метрик портфеля."""
    if equity_df.empty:
        return {}

    equity = equity_df["equity"]
    equity_daily = equity.resample("1D").last().dropna()
    daily_ret = equity_daily.pct_change().dropna()

    equity_start = float(equity.iloc[0])
    equity_end = float(equity.iloc[-1])
    days = max((end - start).days, 1)

    total_return = (equity_end / equity_start - 1.0) if equity_start > 0 else np.nan
    cagr = (equity_end / equity_start) ** (365.0 / days) - 1.0 if equity_start > 0 else np.nan

    ann_vol = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 1 else np.nan
    ann_ret = (1 + float(daily_ret.mean())) ** 252 - 1 if len(daily_ret) > 0 else np.nan
    sharpe = (ann_ret - rf) / ann_vol if (ann_vol and not np.isnan(ann_vol) and ann_vol != 0) else np.nan

    mdd = max_drawdown(equity)
    calmar = (cagr / abs(mdd)) if (mdd and abs(mdd) > 0) else np.nan

    # Метрики по сделкам
    n_trades = len(trades)
    wins = [tr.pnl > 0 for tr in trades]
    win_rate = (sum(wins) / n_trades) if n_trades > 0 else np.nan
    profits = [tr.pnl for tr in trades if tr.pnl > 0]
    losses = [tr.pnl for tr in trades if tr.pnl <= 0]
    avg_win = float(np.mean(profits)) if profits else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    pf = (sum(profits) / abs(sum(losses))) if losses and sum(losses) != 0 else np.nan
    max_wins, max_losses = consecutive_counts(wins)

    exposure = np.nan  # будет заполнено снаружи (процент времени в позиции)

    return {
        "equity_start": equity_start,
        "equity_end": equity_end,
        "total_return": total_return,
        "cagr": cagr,
        "vol_annual": float(ann_vol) if np.isfinite(ann_vol) else np.nan,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": float(mdd) if np.isfinite(mdd) else np.nan,
        "calmar": float(calmar) if np.isfinite(calmar) else np.nan,
        "trades": int(n_trades),
        "win_rate": float(win_rate) if np.isfinite(win_rate) else np.nan,
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(pf) if np.isfinite(pf) else np.nan,
        "max_consecutive_wins": int(max_wins),
        "max_consecutive_losses": int(max_losses),
        "exposure": exposure,  # временно
    }


# ============================== ВИЗУАЛИЗАЦИИ ==============================

def plot_equity(equity_df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(10, 4.5))
    equity_df["equity"].plot()
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_price_signals_single(
    df: pd.DataFrame, trades: List[Trade], out_path: str, ticker: str
) -> None:
    """График цены одного тикера + маркеры входов/выходов."""
    plt.figure(figsize=(11, 5))
    df["Close"].plot()
    # маркеры вход/выход из трейдов по данному тикеру
    entries_x = [tr.entry_time for tr in trades if tr.ticker == ticker]
    entries_y = [tr.entry_price for tr in trades if tr.ticker == ticker]
    exits_x = [tr.exit_time for tr in trades if tr.ticker == ticker]
    exits_y = [tr.exit_price for tr in trades if tr.ticker == ticker]
    if entries_x:
        plt.scatter(entries_x, entries_y, marker="^")
    if exits_x:
        plt.scatter(exits_x, exits_y, marker="v")
    plt.title(f"{ticker} — Price with Entries/Exits")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================== СОХРАНЕНИЕ ОТЧЁТОВ ==============================

def save_reports(
    out_dir: str,
    equity_df: pd.DataFrame,
    trades: List[Trade],
    metrics: Dict[str, float],
    price_plot_single: Optional[Tuple[str, pd.DataFrame, str]] = None,
) -> None:
    ensure_dir(out_dir)

    equity_path = os.path.join(out_dir, "equity_curve.csv")
    trades_path = os.path.join(out_dir, "trades.csv")
    metrics_path = os.path.join(out_dir, "metrics.json")
    equity_plot_path = os.path.join(out_dir, "plot_equity.png")
    plot_equity(equity_df, equity_plot_path)

    equity_df.to_csv(equity_path)

    # trades.csv
    rows = []
    for tr in trades:
        row = asdict(tr)
        row["pnl"] = tr.pnl
        row["ret"] = tr.ret
        rows.append(row)
    trades_df = pd.DataFrame(rows)
    if not trades_df.empty:
        # упорядочим поля
        cols = ["ticker", "entry_time", "exit_time", "entry_price", "exit_price",
                "qty", "fees_entry", "fees_exit", "pnl", "ret"]
        trades_df = trades_df[cols]
    trades_df.to_csv(trades_path, index=False)

    # metrics.json
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Доп. график цены и сигналов, если один тикер
    if price_plot_single is not None:
        ticker, df, out_name = price_plot_single
        out_path = os.path.join(out_dir, out_name)
        plot_price_signals_single(df, trades, out_path, ticker)


# ============================== CLI / MAIN ==============================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Бэктест простой стратегии (один файл).")
    # Данные
    parser.add_argument("--ticker", type=str, default="AAPL",
                        help="Тикер или список через запятую (например, AAPL,MSFT или EURUSD=X).")
    parser.add_argument("--start", type=str, default=None, help="Дата начала (YYYY-MM-DD). По умолчанию ~5 лет назад.")
    parser.add_argument("--end", type=str, default=None, help="Дата окончания (YYYY-MM-DD).")
    parser.add_argument("--interval", type=str, default="1d", choices=["1d", "1h", "30m", "15m"],
                        help="Интервал данных. По умолчанию 1d.")

    # Стратегия
    parser.add_argument("--strategy", type=str, default="sma_cross", choices=["sma_cross", "rsi_revert"],
                        help="Выбор стратегии.")
    parser.add_argument("--fast", type=int, default=10, help="Быстрый период SMA для sma_cross.")
    parser.add_argument("--slow", type=int, default=30, help="Медленный период SMA для sma_cross.")
    parser.add_argument("--rsi_period", type=int, default=14, help="Период RSI для rsi_revert.")
    parser.add_argument("--rsi_buy", type=float, default=30.0, help="Уровень RSI для входа (rsi_revert).")
    parser.add_argument("--rsi_exit", type=float, default=50.0, help="Уровень RSI для выхода (rsi_revert).")

    # Исполнение/риск
    parser.add_argument("--initial_capital", type=float, default=10_000.0, help="Начальный капитал.")
    parser.add_argument("--risk_pct", type=float, default=100.0, help="Процент капитала на тикер (fixed).")
    parser.add_argument("--position", type=str, default="fixed", choices=["fixed", "atr"],
                        help="Метод расчёта размера позиции.")
    parser.add_argument("--risk_budget", type=float, default=1_000.0,
                        help="Риск-бюджет на тикер для позиционирования ATR.")
    parser.add_argument("--atr_period", type=int, default=14, help="Период ATR.")
    parser.add_argument("--atr_k", type=float, default=1.0, help="Множитель ATR в позиционировании (ATR mode).")
    parser.add_argument("--stop_atr_k", type=float, default=None,
                        help="Стоп-лосс как entry_price - k*ATR (опционально).")
    parser.add_argument("--take_profit_perc", type=float, default=None,
                        help="Тейк-профит как +perc от входа, например 0.1 = 10%% (опционально).")
    parser.add_argument("--allow_leverage", action="store_true", help="Разрешить отрицательный кэш (плечо).")

    # Издержки
    parser.add_argument("--fee_perc", type=float, default=0.0, help="Комиссия сделки, доля (на вход и выход).")
    parser.add_argument("--slippage_perc", type=float, default=0.0, help="Слиппедж, доля (ухудшает цену).")

    # Портфель/веса
    parser.add_argument("--weights", type=str, default=None,
                        help="Весы тикеров, например 'AAPL=0.6,MSFT=0.4'. Если не заданы — равные.")

    # Отчёты/прочее
    parser.add_argument("--out", type=str, default="./backtest_out", help="Папка для отчётов.")
    parser.add_argument("--rf", type=float, default=0.0, help="Безрисковая ставка для Sharpe (в долях).")
    parser.add_argument("--loglevel", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def configure_logging(out_dir: str, level: str) -> logging.Logger:
    ensure_dir(out_dir)
    logger = logging.getLogger("backtest")
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()

    # Консоль
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level))
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # Файл
    fh = logging.FileHandler(os.path.join(out_dir, "run.log"), encoding="utf-8")
    fh.setLevel(getattr(logging, level))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def main():
    args = parse_args()
    logger = configure_logging(args.out, args.loglevel)

    # Подготовка параметров по умолчанию по датам
    if not args.start:
        # последние ~5 лет от сегодня
        end_dt = pd.Timestamp.today().normalize()
        start_dt = end_dt - pd.Timedelta(days=365 * 5)
        args.start = start_dt.strftime("%Y-%m-%d")
        if not args.end:
            args.end = end_dt.strftime("%Y-%m-%d")
    start_str = args.start
    end_str = args.end

    # Список тикеров
    tickers = [t.strip() for t in args.ticker.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("Укажите хотя бы один тикер через --ticker")

    # Веса
    weights = parse_weights(args.weights, tickers, logger)

    # Загрузка данных
    data = load_data(tickers, start_str, end_str, args.interval, logger)

    # Генерация сигналов-режимов (1 — хотим быть в позиции; 0 — вне)
    signals: Dict[str, pd.Series] = {}
    for t in data:
        df = data[t]
        if args.strategy == "sma_cross":
            sig = signals_sma_cross(df, args.fast, args.slow)
        else:
            sig = signals_rsi_revert(df, args.rsi_period, args.rsi_buy, args.rsi_exit)
        # после вычисления сигналов — приведём индекс к datetime, удалим явные NaN
        sig = sig.reindex(df.index).fillna(0).astype(int)
        signals[t] = sig

    # Симуляция
    equity_df, trades, invested = simulate_portfolio(
        data=data,
        signals=signals,
        weights=weights,
        initial_capital=args.initial_capital,
        fee_perc=args.fee_perc,
        slippage_perc=args.slippage_perc,
        position_mode=args.position,
        risk_pct=args.risk_pct,
        risk_budget=args.risk_budget,
        atr_period=args.atr_period,
        atr_k=args.atr_k,
        stop_atr_k=args.stop_atr_k,
        take_profit_perc=args.take_profit_perc,
        allow_leverage=args.allow_leverage,
        logger=logger
    )

    # Метрики
    start_ts = equity_df.index[0]
    end_ts = equity_df.index[-1]
    metrics = compute_metrics(equity_df, trades, rf=args.rf, start=start_ts, end=end_ts)
    # Выставим Exposure (доля времени в позиции)
    metrics["exposure"] = float(invested.mean()) if len(invested) else np.nan

    # Консольный вывод
    period_str = f"{start_ts.strftime('%Y-%m-%d')} — {end_ts.strftime('%Y-%m-%d')}"
    strategy_params = ""
    if args.strategy == "sma_cross":
        strategy_params = f"(fast={args.fast}, slow={args.slow})"
    else:
        strategy_params = f"(rsi_period={args.rsi_period}, buy={args.rsi_buy}, exit={args.rsi_exit})"

    print("========== Backtest Summary ==========")
    print(f"Ticker(s)      : {', '.join(tickers)}")
    print(f"Period         : {period_str}")
    print(f"Strategy       : {args.strategy} {strategy_params}")
    print(f"Initial Capital: {metrics.get('equity_start', float(args.initial_capital)):.2f}")
    print("--------------------------------------")
    def pct(x): return f"{(x*100):.1f}%" if x is not None and np.isfinite(x) else "n/a"
    print(f"Total Return   : {pct(metrics.get('total_return', np.nan))}")
    print(f"CAGR           : {pct(metrics.get('cagr', np.nan))}")
    print(f"Sharpe (rf={args.rf:.2%}) : {metrics.get('sharpe', float('nan')):.2f}")
    vol = metrics.get("vol_annual", np.nan)
    print(f"Volatility     : {pct(vol if np.isfinite(vol) else np.nan)}")
    print(f"Max Drawdown   : {pct(metrics.get('max_drawdown', np.nan))}")
    calmar = metrics.get("calmar", np.nan)
    print(f"Calmar         : {calmar:.2f}" if np.isfinite(calmar) else "Calmar         : n/a")
    print(f"Trades         : {metrics.get('trades', 0)} | Win rate: {pct(metrics.get('win_rate', np.nan))} | PF: {metrics.get('profit_factor', float('nan')):.2f}")
    print(f"Avg Win / Loss : {metrics.get('avg_win', 0.0):.2f} / {metrics.get('avg_loss', 0.0):.2f}")
    print(f"Exposure       : {pct(metrics.get('exposure', np.nan))}")
    print("======================================")
    print(f"Files saved to : {os.path.abspath(args.out)}")

    # Сохранение отчётов
    price_plot_single = None
    if len(tickers) == 1:
        t = tickers[0]
        price_plot_single = (t, data[t], "plot_price_signals.png")

    save_reports(
        out_dir=args.out,
        equity_df=equity_df,
        trades=trades,
        metrics=metrics,
        price_plot_single=price_plot_single
    )


if __name__ == "__main__":
    main()