"""
=======================================================
  STOCK PORTFOLIO TRACKER & OPTIMIZER
  Resume Project | Financial Analysis & Portfolio Mgmt
  Stocks: AAPL, TSLA, MSFT, AMZN, GOOGL
  Portfolio Value: $100,000
=======================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────
TICKERS       = ["AAPL", "TSLA", "MSFT", "AMZN", "GOOGL"]
PORTFOLIO_VAL = 100_000          # Starting portfolio value
RISK_FREE     = 0.05             # 5% annual risk-free rate (T-bill approx)
PERIOD        = "2y"             # 2 years of historical data
EQUAL_WEIGHT  = 1 / len(TICKERS) # Equal weight = 20% each

# ── 1. GENERATE REALISTIC PRICE DATA ────────────────
# (In production: replace with yf.download(TICKERS, period=PERIOD) )
print("\n📥 Generating realistic historical price data...")
np.random.seed(42)
n_days = 504  # ~2 years of trading days
dates = pd.bdate_range(end=pd.Timestamp("2025-05-16"), periods=n_days)

# Realistic annualized return / vol parameters per stock
params = {
    "AAPL":  (0.22, 0.28, 182.0),
    "TSLA":  (0.15, 0.68, 245.0),
    "MSFT":  (0.20, 0.24, 375.0),
    "AMZN":  (0.25, 0.32, 178.0),
    "GOOGL": (0.18, 0.26, 140.0),
}
price_data = {}
for t, (mu, sigma, s0) in params.items():
    daily_mu  = mu / 252
    daily_sig = sigma / np.sqrt(252)
    shocks = np.random.normal(daily_mu, daily_sig, n_days)
    price_data[t] = s0 * np.exp(np.cumsum(shocks))

prices = pd.DataFrame(price_data, index=dates)
print(f"   ✅ {len(prices)} trading days ({prices.index[0].date()} → {prices.index[-1].date()}")

# ── 2. CALCULATE DAILY RETURNS ───────────────────────
returns = prices.pct_change().dropna()
ann_returns = returns.mean() * 252        # Annualized mean returns
ann_cov     = returns.cov() * 252         # Annualized covariance matrix
ann_vol     = returns.std() * np.sqrt(252) # Annualized volatility

# ── 3. PORTFOLIO METRICS FUNCTION ───────────────────
def portfolio_metrics(weights, ann_returns, ann_cov):
    w = np.array(weights)
    ret = np.dot(w, ann_returns)
    vol = np.sqrt(np.dot(w.T, np.dot(ann_cov, w)))
    sharpe = (ret - RISK_FREE) / vol
    return ret, vol, sharpe

# ── 4. CURRENT EQUAL-WEIGHT PORTFOLIO ───────────────
eq_weights = [EQUAL_WEIGHT] * len(TICKERS)
eq_ret, eq_vol, eq_sharpe = portfolio_metrics(eq_weights, ann_returns, ann_cov)

shares = {t: round((PORTFOLIO_VAL * EQUAL_WEIGHT) / prices[t].iloc[-1], 2) for t in TICKERS}
current_prices = {t: round(prices[t].iloc[-1], 2) for t in TICKERS}
alloc_values   = {t: round(shares[t] * current_prices[t], 2) for t in TICKERS}

# ── 5. SHARPE-OPTIMIZED PORTFOLIO ───────────────────
print("\n🔧 Running Sharpe Ratio optimization...")

def neg_sharpe(weights):
    _, _, sharpe = portfolio_metrics(weights, ann_returns, ann_cov)
    return -sharpe

constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = [(0.05, 0.60)] * len(TICKERS)   # 5% min, 60% max per stock

result = minimize(
    neg_sharpe, eq_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)
opt_weights = result.x
opt_ret, opt_vol, opt_sharpe = portfolio_metrics(opt_weights, ann_returns, ann_cov)
print(f"   ✅ Optimization complete (Sharpe: {opt_sharpe:.2f})")

# ── 6. MONTE CARLO SIMULATION ────────────────────────
print("\n🎲 Running Monte Carlo simulation (5,000 portfolios)...")
N_SIM = 5_000
mc_returns, mc_vols, mc_sharpes, mc_weights = [], [], [], []

for _ in range(N_SIM):
    w = np.random.dirichlet(np.ones(len(TICKERS)))
    r, v, s = portfolio_metrics(w, ann_returns, ann_cov)
    mc_returns.append(r)
    mc_vols.append(v)
    mc_sharpes.append(s)
    mc_weights.append(w)

mc_returns  = np.array(mc_returns)
mc_vols     = np.array(mc_vols)
mc_sharpes  = np.array(mc_sharpes)

# ── 7. PRINT RESULTS REPORT ──────────────────────────
print("\n" + "="*55)
print("  PORTFOLIO TRACKER — RESULTS REPORT")
print("="*55)

print("\n📊 INDIVIDUAL STOCK METRICS (Annualized):")
print(f"{'Ticker':<8} {'Price':>8} {'Ann.Return':>12} {'Ann.Volatility':>15} {'Allocation':>12}")
print("-"*55)
for t in TICKERS:
    print(f"{t:<8} ${current_prices[t]:>7} {ann_returns[t]:>11.1%} {ann_vol[t]:>14.1%} ${alloc_values[t]:>11,.0f}")

print("\n💼 EQUAL-WEIGHT PORTFOLIO (Baseline):")
print(f"   Expected Annual Return : {eq_ret:.2%}")
print(f"   Annual Volatility      : {eq_vol:.2%}")
print(f"   Sharpe Ratio           : {eq_sharpe:.2f}")

print("\n🚀 OPTIMIZED PORTFOLIO (Max Sharpe):")
print(f"{'Ticker':<8} {'Weight':>8} {'$ Allocated':>12}")
print("-"*32)
for t, w in zip(TICKERS, opt_weights):
    print(f"{t:<8} {w:>7.1%} ${w * PORTFOLIO_VAL:>11,.0f}")
print("-"*32)
print(f"   Expected Annual Return : {opt_ret:.2%}")
print(f"   Annual Volatility      : {opt_vol:.2%}")
print(f"   Sharpe Ratio           : {opt_sharpe:.2f}")

print("\n📈 SHARPE IMPROVEMENT:")
improvement = ((opt_sharpe - eq_sharpe) / abs(eq_sharpe)) * 100
print(f"   {eq_sharpe:.2f} → {opt_sharpe:.2f}  ({improvement:+.1f}%)")

# ── 8. REBALANCING RECOMMENDATION ───────────────────
print("\n⚖️  REBALANCING RECOMMENDATION:")
for t, ew, ow in zip(TICKERS, eq_weights, opt_weights):
    diff = ow - ew
    action = "BUY  +" if diff > 0 else "SELL "
    print(f"   {t:<5} {action}{abs(diff):.1%}  →  target {ow:.1%}")

# ── 9. PLOT DASHBOARD ────────────────────────────────
print("\n📊 Generating charts...")

fig = plt.figure(figsize=(18, 12), facecolor="#0f1117")
fig.suptitle("Stock Portfolio Tracker & Optimizer  |  $100,000 Portfolio",
             fontsize=16, fontweight="bold", color="white", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)
ax_price  = fig.add_subplot(gs[0, :2])
ax_weight = fig.add_subplot(gs[0, 2])
ax_mc     = fig.add_subplot(gs[1, :2])
ax_bar    = fig.add_subplot(gs[1, 2])

COLORS = ["#00d4ff", "#ff6b6b", "#51cf66", "#ffd43b", "#cc5de8"]
bg = "#0f1117"; panel = "#1a1d27"; text = "#e0e0e0"; grid_c = "#2a2d3a"

for ax in [ax_price, ax_weight, ax_mc, ax_bar]:
    ax.set_facecolor(panel)
    ax.tick_params(colors=text, labelsize=9)
    ax.spines[:].set_color(grid_c)

# -- Normalized price chart
norm = prices / prices.iloc[0] * 100
for i, t in enumerate(TICKERS):
    ax_price.plot(norm.index, norm[t], color=COLORS[i], linewidth=1.5, label=t)
ax_price.set_title("Normalized Price Performance (Base = 100)", color=text, fontsize=11, pad=8)
ax_price.set_ylabel("Index", color=text)
ax_price.legend(facecolor=panel, labelcolor=text, fontsize=9, ncol=5)
ax_price.grid(True, color=grid_c, linewidth=0.4)

# -- Weight comparison bar chart
x = np.arange(len(TICKERS))
ax_weight.bar(x - 0.2, [EQUAL_WEIGHT]*len(TICKERS), 0.35, label="Equal", color="#4a4e69", alpha=0.9)
ax_weight.bar(x + 0.2, opt_weights, 0.35, label="Optimized", color=COLORS, alpha=0.9)
ax_weight.set_xticks(x); ax_weight.set_xticklabels(TICKERS, color=text, fontsize=9)
ax_weight.set_title("Weight Comparison", color=text, fontsize=11, pad=8)
ax_weight.set_ylabel("Weight", color=text)
ax_weight.legend(facecolor=panel, labelcolor=text, fontsize=9)
ax_weight.grid(axis="y", color=grid_c, linewidth=0.4)
ax_weight.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

# -- Monte Carlo efficient frontier
sc = ax_mc.scatter(mc_vols, mc_returns, c=mc_sharpes, cmap="plasma",
                   s=4, alpha=0.5, rasterized=True)
plt.colorbar(sc, ax=ax_mc, label="Sharpe Ratio").ax.yaxis.label.set_color(text)
ax_mc.scatter(eq_vol, eq_ret, marker="D", s=150, color="#00d4ff", zorder=5, label=f"Equal-Weight (SR={eq_sharpe:.2f})")
ax_mc.scatter(opt_vol, opt_ret, marker="*", s=300, color="#ffd43b", zorder=6, label=f"Optimized (SR={opt_sharpe:.2f})")
ax_mc.set_title("Monte Carlo Simulation — Efficient Frontier (5,000 Portfolios)", color=text, fontsize=11, pad=8)
ax_mc.set_xlabel("Annual Volatility", color=text)
ax_mc.set_ylabel("Annual Return", color=text)
ax_mc.legend(facecolor=panel, labelcolor=text, fontsize=9)
ax_mc.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax_mc.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax_mc.grid(True, color=grid_c, linewidth=0.4)

# -- Sharpe & Return bar per stock
metrics_df = pd.DataFrame({
    "Return": ann_returns.values,
    "Sharpe": [(ann_returns[t] - RISK_FREE) / ann_vol[t] for t in TICKERS]
}, index=TICKERS)
x2 = np.arange(len(TICKERS))
ax_bar.bar(x2 - 0.2, metrics_df["Return"], 0.35, label="Ann. Return", color=COLORS, alpha=0.9)
ax_bar.bar(x2 + 0.2, metrics_df["Sharpe"] / 5, 0.35, label="Sharpe (/5)", color=COLORS, alpha=0.5, hatch="//")
ax_bar.set_xticks(x2); ax_bar.set_xticklabels(TICKERS, color=text, fontsize=9)
ax_bar.set_title("Return & Sharpe per Stock", color=text, fontsize=11, pad=8)
ax_bar.legend(facecolor=panel, labelcolor=text, fontsize=9)
ax_bar.grid(axis="y", color=grid_c, linewidth=0.4)
ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

plt.savefig("/mnt/user-data/outputs/portfolio_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=bg)
print("   ✅ Dashboard saved → portfolio_dashboard.png")
print("\n✅ Project complete! Check portfolio_dashboard.png for charts.\n")
