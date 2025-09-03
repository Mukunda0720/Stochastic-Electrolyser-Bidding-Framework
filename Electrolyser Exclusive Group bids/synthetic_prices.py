"""
synthetic_prices.py
-------------------
Generator for synthetic day-ahead electricity prices
with:
    • Daily seasonality  (sine wave)
    • Slow linear trend  (drift)
    • AR(1) residual noise (gives serial correlation)
    • Stochastic spikes  (positive and optional negative)
    • Separate forecast series with controllable error

10 user-facing dials:
    mu, A, beta,               # level, diurnal swing, trend
    sigma, phi,                # noise volatility + persistence
    p_pos, S_pos,              # positive-spike frequency & size
    p_neg, S_neg,              # negative-spike     »       »
    sigma_f                    # forecast accuracy

Author: <your name>, 2025
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Main generator
# ------------------------------------------------------------------
def synth_prices_ar1(
    hours: int,
    *,
    mu: float = 80,
    A: float = 30,
    beta: float = 0.0,
    sigma: float = 40,
    phi: float = 0.7,
    p_pos: float = 0.02,
    S_pos: float = 300,
    p_neg: float = 0.00,
    S_neg: float = 100,
    sigma_f: float = 15,
    seed: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Return two pandas Series of length `hours`:
        actual_prices  – simulated clearing prices
        forecast_prices – day-ahead forecast with error σ_f

    Parameters
    ----------
    hours   : int
        Number of hourly steps to simulate.
    mu      : float
        Long-run mean price (€/MWh).
    A       : float
        Amplitude of the daily sine wave (€/MWh).
    beta    : float
        Linear drift in €/MWh per hour (positive = upward trend).
    sigma   : float
        Standard deviation of high-frequency noise.
    phi     : float
        AR(1) persistence (0→white noise, ≈0.8 gives strong autocorr).
    p_pos   : float
        Probability an hour is a positive spike.
    S_pos   : float
        Mean height of positive spikes above baseline.
    p_neg   : float
        Probability an hour is a negative spike.
    S_neg   : float
        Mean depth of negative spikes below baseline (positive value).
    sigma_f : float
        Std-dev of forecast error added to each hour.
    seed    : int | None
        RNG seed for reproducibility.

    Returns
    -------
    actual_prices   : pd.Series (€/MWh)
    forecast_prices : pd.Series (€/MWh)
    """
    rng = np.random.default_rng(seed)

    # ---- 1 · Baseline: mean + daily sine + linear drift ----
    t = np.arange(hours)
    # Correctly models daily cycle with peak in the afternoon
    daily_cycle = np.sin(2 * np.pi * ((t - 6) % 24) / 24)
    baseline = mu + A * daily_cycle + beta * t

    # ---- 2 · AR(1) residual noise ε_t = φ ε_{t-1} + η_t ----
    eps = np.zeros(hours)
    # Scale η_t so overall residual std is σ
    eta_std = sigma * np.sqrt(1 - phi**2) if phi < 1 else sigma
    for i in range(1, hours):
        eps[i] = phi * eps[i - 1] + rng.normal(0, eta_std)

    # ---- 3 · Random scarcity & curtailment spikes ----
    spikes = np.zeros(hours)
    U = rng.random(hours)
    pos = U < p_pos                                    # positive spike hours
    neg = (U >= p_pos) & (U < p_pos + p_neg)           # negative spike hours
    spikes[pos] = rng.normal(S_pos, 0.5 * S_pos, pos.sum())
    spikes[neg] = -np.abs(rng.normal(S_neg, 0.5 * S_neg, neg.sum()))

    # ---- 4 · Compose synthetic actual price ----
    actual = baseline + eps + spikes

    # ---- 5 · Forecast = actual + independent error ----
    forecast = actual + rng.normal(0, sigma_f, hours)

    # ---- 6 · Wrap in Series with a generic hourly index ----
    # Note: The start date is arbitrary here; the main script will align it.
    idx = pd.date_range("2025-01-01", periods=hours, freq="H")
    return pd.Series(actual, index=idx, name="actual"), pd.Series(
        forecast, index=idx, name="forecast"
    )


# ------------------------------------------------------------------
# Example usage: simulate one synthetic year
# ------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    HOURS_PER_YEAR = 24 * 365
    actual, forecast = synth_prices_ar1(
        HOURS_PER_YEAR,
        mu=60,
        A=35,
        beta=0.01,
        sigma=35,
        phi=0.8,
        p_pos=0.015,
        S_pos=350,
        p_neg=0.03,
        S_neg=120,
        sigma_f=10,
        seed=42,
    )

    # Quick sanity check
    print(f"Lag-1 autocorr : {np.corrcoef(actual[:-1], actual[1:])[0, 1]:.3f}")
    print(f"Std-dev (€/MWh): {actual.std():.1f}")
    print(f"Hours > €450   : {(actual > 450).sum()}")
    print(f"Hours < €0     : {(actual < 0).sum()}")

    # Optional: plot first month if matplotlib is installed
    plt.figure(figsize=(15, 7))
    actual[:24 * 31].plot(label="Actual Price", color='blue')
    forecast[:24 * 31].plot(label="Forecast Price", color='orange', alpha=0.7, linestyle='--')
    plt.ylabel("Price (€/MWh)")
    plt.title("Synthetic Prices – First Month")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
