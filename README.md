# investments_project

Проект строит и проверяет оптимальный портфель на основе алгоритмов из курса:
mean-variance optimization, efficient frontier, CML/CAPM, VaR и Expected
Shortfall. Основная стратегия — `Opt No-Short`, то есть оптимальный long-only
портфель без коротких продаж.

## Что делает код

- Загружает дневные данные ETF через `yfinance`.
- Считает доходности, ковариации, beta, VaR, ES, drawdown и Sharpe ratio.
- Решает задачу оптимизации портфеля через `scipy.optimize`:
  `max w^T mu` при ограничениях на риск, сумму весов и short/no-short режим.
- Подбирает гиперпараметры через grid search:
  обучающее окно, частоту ребалансировки и риск-лимит.
- Проводит rolling backtest без использования будущей информации.
- Сравнивает стратегию с `Equal Weight` и `SPY`.
- Генерирует отчет, CSV-таблицы и графики в `results/`.

## Активы

Используются ETF: `SPY`, `QQQ`, `IWM`, `XLF`, `XLK`, `XLE`, `XLV`, `TLT`,
`GLD`. Такой набор покрывает широкий рынок, технологический рост, малую
капитализацию, несколько секторов, облигации и золото.

## Структура

- `generate_practical_part.py` — основной скрипт расчетов и генерации отчета.
- `results/practical_part.md` — готовая практическая часть.
- `results/metrics_test.csv` — итоговые метрики на тестовом периоде.
- `results/calibration_grid.csv` — вся сетка калибровки гиперпараметров.
- `results/top_calibration_configs.csv` — лучшие конфигурации.
- `results/weights_no_short.csv` — веса основной стратегии.
- `results/figures/` — графики цен, корреляций, frontier/CML, весов,
  накопленной доходности и просадок.

## Запуск

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python generate_practical_part.py
```

После запуска обновляются таблицы, графики и `results/practical_part.md`.
