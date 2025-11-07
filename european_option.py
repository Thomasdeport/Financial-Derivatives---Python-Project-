import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm


class EuropeanOption:
    """European option pricer under Black–Scholes with strategy support."""

    def __init__(self, S, K, T, r, sigma, option_type="call", derivatives=None):
        self.S = S
        self.K = np.atleast_1d(K)
        self.T = np.atleast_1d(T)
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.derivatives = derivatives.lower() if derivatives else None

    # === Core Black-Scholes ===

    def _d1(self, S, K, T):
        return (np.log(S / K) + (self.r + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T))

    def _d2(self, S, K, T):
        return self._d1(S, K, T) - self.sigma * np.sqrt(T)

    def _bs_price(self, S, K, T, option_type):
        d1, d2 = self._d1(S, K, T), self._d2(S, K, T)
        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _greeks(self, S, K, T, option_type):
        d1, d2 = self._d1(S, K, T), self._d2(S, K, T)
        pdf = norm.pdf(d1)
        if option_type == "call":
            delta = norm.cdf(d1)
            rho = K * T * np.exp(-self.r * T) * norm.cdf(d2) / 100
            theta = (-S * pdf * self.sigma / (2 * np.sqrt(T))
                     - self.r * K * np.exp(-self.r * T) * norm.cdf(d2)) / 365
        else:
            delta = -norm.cdf(-d1)
            rho = -K * T * np.exp(-self.r * T) * norm.cdf(-d2) / 100
            theta = (-S * pdf * self.sigma / (2 * np.sqrt(T))
                     + self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)) / 365
        gamma = pdf / (S * self.sigma * np.sqrt(T))
        vega = S * pdf * np.sqrt(T) / 100
        return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

    # === Multi-leg Strategies ===

    def _compose_strategy(self, S):
        strategies = {
            "straddle": (1, [("call", 0, +1), ("put", 0, +1)]),
            "strangle": (2, [("call", 1, +1), ("put", 0, +1)]),
            "bull_call_spread": (2, [("call", 0, +1), ("call", 1, -1)]),
            "bear_put_spread": (2, [("put", 0, +1), ("put", 1, -1)]),
            "collar": (2, [("put", 0, +1), ("call", 1, -1)]),
            "butterfly": (3, [("call", 0, +1), ("call", 1, -2), ("call", 2, +1)]),
            "iron_condor": (4, [
                ("put", 0, +1), ("put", 1, -1),
                ("call", 2, +1), ("call", 3, -1)
            ])
        }

        if self.derivatives not in strategies:
            raise ValueError(f"Unknown strategy '{self.derivatives}'.")

        required, structure = strategies[self.derivatives]
        if len(self.K) < required:
            raise ValueError(
                f"Strategy '{self.derivatives}' requires {required} strikes, got {len(self.K)}."
            )

        results = []
        for opt_type, k_idx, qty in structure:
            K = self.K[k_idx]
            greeks = self._greeks(S, K, self.T[0], opt_type)
            results.append({k: qty * v for k, v in greeks.items()})

        # Sum Greeks across legs
        df = pd.DataFrame(results)
        summary = df.sum().to_dict()
        return summary

    # === Greeks plotting ===

    def plot_greeks(self, x_axis: str = "S", greek_list=None, n_points: int = 50):
        """
        Plot the selected Greeks as a function of a chosen variable.
        Works for single options and composed products.
        x_axis: 'S' | 'sigma' | 'T'
        """
        if greek_list is None:
            greek_list = ["Delta", "Gamma", "Vega", "Theta", "Rho"]

        if x_axis == "S":
            x_vals = np.linspace(0.5 * self.S, 1.5 * self.S, n_points)
        elif x_axis == "sigma":
            x_vals = np.linspace(0.05, 1.0, n_points)
        elif x_axis == "T":
            x_vals = np.linspace(0.01, self.T[0] * 2, n_points)
        else:
            raise ValueError("x_axis must be one of: 'S', 'sigma', 'T'")

        data = []
        for x in x_vals:
            # Update context
            if x_axis == "S":
                S, sigma, T = x, self.sigma, self.T[0]
            elif x_axis == "sigma":
                S, sigma, T = self.S, x, self.T[0]
                self.sigma = x  # temporarily adjust volatility
            elif x_axis == "T":
                S, sigma, T = self.S, self.sigma, x

            # Compute total Greeks
            if self.derivatives:
                summary = self._compose_strategy(S)
            else:
                summary = self._greeks(S, self.K[0], T, self.option_type)
            data.append(summary)

        df = pd.DataFrame(data)
        df[x_axis] = x_vals

        # Plot all requested Greeks
        fig = go.Figure()
        for greek in greek_list:
            if greek in df.columns:
                fig.add_trace(go.Scatter(x=df[x_axis], y=df[greek],
                                         mode="lines", name=greek))
        fig.update_layout(
            title=f"Total Greeks vs {x_axis.upper()} ({self.derivatives or self.option_type})",
            xaxis_title=x_axis.upper(),
            yaxis_title="Greek Value",
            template="plotly_dark",
            legend_title="Greek"
        )
        fig.show()
