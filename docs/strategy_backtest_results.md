# Exploratory Trading Strategy Backtest — Results

## Experiment setup

- Ticker: RELIANCE.NS
- Evaluation period: 2008-01-17 to 2022-06-14
- Sessions: 3,549
- Forecast-driven rule: long at the target session open only when
  forecast-implied upside to the close exceeds assumed round-trip costs;
  exit at the close; otherwise remain in cash.
- Main cost scenario: 10 bps per side, or 20 bps per round trip
- Cost scenarios: 0, 5, 10 and 20 bps per side
- Validation: historical development-period experiment

## Overall results at 10 bps per side

| Strategy | Exposure | Gross total return | Net total return | Annualized net return | Net Sharpe (zero risk-free rate) | Maximum net drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Buy-and-hold | 100.00% | 276.90% | 276.12% | 9.86% | 0.4477 | -67.59% |
| Cash | 0.00% | 0.00% | 0.00% | 0.00% | N/A | 0.00% |
| Random Walk signal | 25.30% | -45.28% | -90.95% | -15.68% | -0.9259 | -91.80% |
| Ridge — all features | 31.70% | -29.44% | -92.59% | -16.87% | -0.9407 | -93.91% |
| Ridge — returns only | 30.94% | -37.72% | -93.10% | -17.29% | -0.9670 | -94.96% |
| Random Forest signal | 30.01% | -54.86% | -94.66% | -18.78% | -1.0817 | -95.26% |
| Drift signal | 29.59% | -63.05% | -95.50% | -19.76% | -1.1257 | -95.87% |
| Ridge — trend only | 29.39% | -65.81% | -95.77% | -20.12% | -1.1546 | -96.21% |
| Ridge — volatility only | 29.56% | -69.01% | -96.22% | -20.75% | -1.2031 | -96.57% |
| Decision Tree signal | 28.83% | -73.06% | -96.54% | -21.24% | -1.2169 | -96.71% |
| Always-long intraday | 100.00% | -98.58% | -100.00% approximately | -55.40% | -2.4932 | -100.00% approximately |

The buy-and-hold benchmark buys at the first evaluation session's open,
remains invested through overnight and intraday moves, and exits at the
last session's close. It pays one entry and one exit cost over the period.

The cash benchmark assumes zero interest. The always-long intraday
benchmark enters at every session's open and exits at every close,
paying assumed round-trip costs on each session.

## Net total return sensitivity

| Strategy | 0 bps/side | 5 bps/side | 10 bps/side | 20 bps/side |
|---|---:|---:|---:|---:|
| Buy-and-hold | 276.90% | 276.51% | 276.12% | 275.34% |
| Cash | 0.00% | 0.00% | 0.00% | 0.00% |
| Random Walk signal | -45.28% | -77.74% | -90.95% | -98.51% |
| Ridge — all features | -29.44% | -77.12% | -92.59% | -99.22% |
| Ridge — returns only | -37.72% | -79.25% | -93.10% | -99.24% |
| Random Forest signal | -54.86% | -84.46% | -94.66% | -99.37% |
| Drift signal | -63.05% | -87.09% | -95.50% | -99.45% |
| Ridge — trend only | -65.81% | -87.97% | -95.77% | -99.48% |
| Ridge — volatility only | -69.01% | -89.17% | -96.22% | -99.54% |
| Decision Tree signal | -73.06% | -90.34% | -96.54% | -99.56% |
| Always-long intraday | -98.58% | -99.96% | -100.00% approximately | -100.00% approximately |

The cost numbers are sensitivity assumptions, not verified Indian
brokerage charges or a complete model of fees, taxes, slippage and
market impact. Cost-adjusted returns use simplified deductions.

## Findings

1. Every forecast-driven strategy had a negative gross cumulative return
   before transaction costs under the tested rule.
2. At 10 bps per side, all forecast-driven strategies had large negative
   net cumulative returns and large drawdowns.
3. The zero-interest cash benchmark outperformed every forecast-driven
   strategy.
4. Buy-and-hold produced a positive cumulative return of approximately
   276.12% net under the stated two-transaction cost assumption, but
   it has different exposure and risk from the intraday strategies.
5. Increasing assumed costs worsened the strategy outcomes substantially.
6. These results do not establish that the forecasts have predictive
   or economic value.

## Research decision

Mark this strategy rule as unsuccessful in the tested development period.
Do not continue optimizing its threshold against this same dataset.

Do not describe these results as independent or out-of-sample proof of
future performance. Multiple models and strategies have already been
examined, and the previously evaluated 2022-06-15 to 2026-10-23 period
must not be repeatedly reused for tuning.

Before another strategy is tested, state a specific economic hypothesis,
position and exit rules, realistic costs, and a predeclared evaluation
method. If no defensible hypothesis is available, stop strategy expansion
rather than searching indefinitely for a positive result.

## Limitations

Costs are simplified assumptions rather than verified broker charges.
Cash earns zero interest in this comparison. Buy-and-hold and intraday
strategies have different exposures and are not perfectly risk-matched.
Forecast-error metrics and historical strategy returns do not establish
future profitability.
