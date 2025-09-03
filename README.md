# Python-бэктест стратегий с визуализацией сделок

`backtest.py` — это лёгкий бэктестер для акций/ETF/форекс/крипто на исторических данных Yahoo Finance (через `yfinance`). Поддерживает 2 стратегии из коробки, комиссии/слиппедж, T+1 исполнение по цене `Open`, позиционирование `fixed/ATR`, стоп-лосс по ATR, отчёты (CSV/JSON) и графики (PNG).

---

## 🚀 Возможности

* Загрузка котировок (акции, ETF, форекс, крипто) через `yfinance`.
* Интервалы: `1d`, `1h`, `30m`, `15m`.
* Портфель из нескольких тикеров с весами (по умолчанию равные).
* Стратегии:

  * **SMA Cross**: вход, когда `SMA(fast) > SMA(slow)`.
  * **RSI Revert**: вход при перепроданности (`RSI < buy`), выход при `RSI > exit`.
* Исполнение сигналов: **на следующей свече** по цене `Open` (T+1).
* Издержки: комиссия и слиппедж.
* Позиционирование: фиксированный процент капитала или риск-менеджмент по ATR.
* Риск-контроль: стоп-лосс по `k * ATR`, тейк-профит в процентах.
* Метрики: Total Return, CAGR, Volatility, Sharpe, Max Drawdown, Calmar, Win rate, Profit Factor и др.
* Артефакты: `equity_curve.csv`, `trades.csv`, `metrics.json`, `plot_equity.png`, `plot_price_signals.png`.

---

## 🧩 Требования

* Python 3.9+
* Пакеты: `pandas`, `numpy`, `matplotlib`, `yfinance`

Установка зависимостей:

```bash
pip install -U pandas numpy matplotlib yfinance
```

---

## 📦 Быстрый старт

Склонировать репозиторий и запустить:

```bash
git clone https://github.com/FX84/Backtest-Trading-Strategies.git
cd Backtest-Trading-Strategies
python backtest.py
```

По умолчанию:

* Тикер: `AAPL`
* История: последние \~5 лет
* Интервал: `1d`
* Стратегия: `sma_cross` (`fast=10`, `slow=30`)
* Капитал: `10_000`
* Издержки: `0`
* Отчёты в папку: `./backtest_out`

---

## 🔧 Примеры команд

### 1) SMA-cross на дневках (акции)

```bash
python backtest.py --ticker AAPL --start 2018-01-01 \
  --strategy sma_cross --fast 10 --slow 30 \
  --fee_perc 0.001 --slippage_perc 0.0005
```

### 2) RSI-mean-reversion на часовках (форекс)

```bash
python backtest.py --ticker EURUSD=X --start 2022-01-01 --interval 1h \
  --strategy rsi_revert --rsi_period 14 --rsi_buy 30 --rsi_exit 50
```

### 3) Портфель AAPL+MSFT равными долями, стоп 3×ATR

```bash
python backtest.py --ticker AAPL,MSFT --start 2019-01-01 \
  --strategy sma_cross --fast 20 --slow 100 \
  --stop_atr_k 3 --fee_perc 0.001
```

### 4) Кастомные веса портфеля + отчёты в свою папку

```bash
python backtest.py --ticker AAPL,MSFT --weights AAPL=0.6,MSFT=0.4 \
  --start 2020-01-01 --out ./reports/my_test
```

---

## 🧮 Параметры CLI

**Данные**

* `--ticker` — тикер(ы), через запятую (напр. `AAPL,MSFT` или `EURUSD=X`).
* `--start`, `--end` — даты в формате `YYYY-MM-DD`. Если не указано — \~5 лет.
* `--interval` — `1d | 1h | 30m | 15m`.

**Стратегия**

* `--strategy` — `sma_cross | rsi_revert`
* `--fast`, `--slow` — периоды SMA для `sma_cross`.
* `--rsi_period`, `--rsi_buy`, `--rsi_exit` — параметры RSI для `rsi_revert`.

**Исполнение/риск**

* `--initial_capital` — стартовый капитал.
* `--position` — `fixed | atr`.
* `--risk_pct` — % капитала на тикер (режим `fixed`).
* `--risk_budget` — риск-бюджет на тикер (режим `atr`).
* `--atr_period`, `--atr_k` — параметры ATR для позиционирования.
* `--stop_atr_k` — стоп-лосс как `entry - k*ATR`.
* `--take_profit_perc` — тейк-профит, доля (напр. `0.1 = 10%`).
* `--allow_leverage` — разрешить отрицательный кэш (плечо).

**Издержки**

* `--fee_perc` — комиссия (доля) на вход/выход.
* `--slippage_perc` — слиппедж (ухудшение цены).

**Портфель**

* `--weights` — веса, напр. `AAPL=0.6,MSFT=0.4` (нормализуются до 1).

**Отчёты/другое**

* `--out` — директория для результатов (по умолчанию `./backtest_out`).
* `--rf` — безрисковая ставка для Sharpe (в долях, напр. `0.02`).
* `--loglevel` — `DEBUG | INFO | WARNING | ERROR`.

---

## 📁 Выходные файлы

В каталоге `--out` появятся:

* `equity_curve.csv` — кривая капитала и кэш по датам.
* `trades.csv` — журнал сделок: вход/выход, цена, размер, комиссии, PnL, доходность сделки.
* `metrics.json` — метрики производительности и параметры запуска.
* `plot_equity.png` — график кривой капитала.
* `plot_price_signals.png` — цена + точки входа/выхода (если 1 тикер).

---

## 📊 Метрики

* **Total Return**: `(Equity_end / Equity_start - 1)`.
* **CAGR**: `((Equity_end / Equity_start) ** (365 / days) - 1)`.
* **Volatility (annualized)**: `std(daily_returns) * sqrt(252)`.
* **Sharpe**: `(annual_return - rf) / annual_vol`.
* **Max Drawdown** — максимальная просадка кривой капитала.
* **Calmar**: `CAGR / |Max Drawdown|`.
* **Trades**, **Win rate**, **Avg Win / Avg Loss**, **Profit Factor**,
  **Max Consecutive Wins/Losses**, **Exposure** (доля времени в позиции).

---

## ⚙️ Логика исполнения

* Сигнал формируется на свече `t` (по данным до `Close[t]`).
* Сделка исполняется на **следующей свече** (`t+1`) по цене `Open[t+1]` с учётом слиппеджа.
* Комиссия списывается на входе и выходе.
* Позиционирование:

  * `fixed`: % капитала \* вес тикера.
  * `atr`: `qty ≈ risk_budget * weight / (atr_k * ATR)`.
* Стоп/тейк:

  * Стоп: `entry - stop_atr_k * ATR_entry`.
  * Тейк: `entry * (1 + take_profit_perc)`.
  * При одновременном срабатывании приоритет у стопа.

---

## 🧱 Ограничения и примечания

* Данные Yahoo Finance могут содержать пропуски/корректировки; для внутридневных баров история ограничена.
* Моделирование максимально упрощено (нет комиссий за перенос, процентов на кэш, проскальзывание — линейное).
* Результаты бэктестов **не гарантируют** будущую доходность.

**Дисклеймер:** Материал предоставлен исключительно в образовательных целях и не является инвестиционной рекомендацией.

---

## 🗂 Структура проекта

```
Backtest-Trading-Strategies/
├─ backtest.py          # основной и единственный код
├─ backtest_out/        # папка с отчётами (создаётся при запуске)
└─ README.md
```

---

## 🔍 Отладка

* Включить подробные логи:

  ```bash
  python backtest.py --loglevel DEBUG ...
  ```
* Если `trades.csv` пустой — проверь параметры стратегии, период и интервал.
* Если возникает ошибка загрузки данных — проверь правильность тикера и соединение с интернетом.
