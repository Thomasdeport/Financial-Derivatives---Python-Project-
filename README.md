# Financial Derivatives - Python Project

This repository presents a **Python-based project on dynamic hedging of vanilla equity options**, with a detailed **PnL decomposition by Greeks** (Delta, Gamma, Theta, Vega, Rho) and an exploration of the **practical limits of the Black–Scholes framework** (discrete rebalancing, volatility changes, market frictions, etc.).

In addition to pricing and hedging aspects, the project includes **econometric analyses** implemented in dedicated Jupyter notebooks.

## Objectives

* Implement and analyze a dynamic hedging strategy for vanilla options.
* Decompose portfolio PnL into Greek-driven components and study hedging errors.
* Assess the impact of discrete hedging, volatility dynamics, and model assumptions.
* Complement the hedging framework with econometric analysis of financial time series.

## Repository Structure

* **`backtester.ipynb`**
  Main notebook of the project. It implements the **backtesting framework** for dynamic hedging strategies and should be considered the core file of the repository.

* **`financial_derivatives_econometrics.ipynb`**
  Notebook dedicated to **econometric analyses**, providing statistical insights related to the project.

* **`market_data_loader/`**
  Utilities for loading and preprocessing market data.

* **`volatility_interpolator/`**
  Tools for volatility interpolation (e.g. implied volatility smile or surface handling) used for other related projects.

* **`econometrics/`**
  Additional modules or resources related to econometric modeling.

* **`data/`**
  Market data (or sample datasets) used in the notebooks.

* **`ml_prediction.py`**
  Optional script introducing a predictive component that can be integrated into the workflow.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Thomasdeport/Financial-Derivatives---Python-Project-.git
cd Financial-Derivatives---Python-Project-
```

### 2. Set up a Python environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
pip install --upgrade pip
```

Install the required dependencies based on the imports used in the notebooks
(`backtester.ipynb` and `financial_derivatives_econometrics.ipynb`, you can easily do it using the `requirements.txt` file using the command zch pip install -r "requirements.txt" ).


### 3. Launch Jupyter

```bash
pip install notebook jupyterlab
jupyter lab
```

Then open:

* `backtester.ipynb`
* `financial_derivatives_econometrics.ipynb`

## Backtesting Framework Overview

The **backtester** is designed to:

1. Load and preprocess market data (spot prices, volatility parameters, interest rates).
2. Price vanilla options and compute Greeks under Black–Scholes-type assumptions.
3. Implement **discrete-time dynamic hedging strategies** (delta hedging and extensions).
4. Produce:

   * Portfolio value trajectories
   * Total PnL
   * Greek-based PnL decomposition
   * Diagnostics on hedging frequency, volatility changes, and modeling assumptions

This allows a clear interpretation of hedging performance and sources of error.

## Backtester Class Architecture (in `backtester.ipynb`)

The notebook is structured around a small set of classes that separate (i) option pricing/Greeks, (ii) hedging/backtesting orchestration, and (iii) scenario extensions (crash, rates, transaction costs, delta–gamma hedging).

### 1) `EuropeanOption`, pricer + Greeks (+ smile support)
`EuropeanOption` is the core instrument class implementing **Black–Scholes pricing** and **Greek calculations**.

Key design points:
- Supports plain European calls/puts, and can also represent **multi-leg strategies** via aggregation (a strategy can be composed from multiple option “legs”).
- Optionally supports an **implied volatility smile surface** using a quadratic parameterization per maturity
  with **linear interpolation in maturity**.
- Greeks are computed in a **sticky-strike** sense ( ** the volatility skew for an option remains unchanged with strike) . 
- Includes utilities for plotting Greek profiles.

Main methods:
- internal BS building blocks (`_d1`, `_d2`, `_bs_price`, `_greeks`)
- smile interpolation helpers (`_interp_smile_coeffs`, `_sigma_eff`)
- strategy aggregation (`_compose_strategy`)
- visualization (`plot_greeks`)

### 2) `OptionHedger`, main backtesting engine
`OptionHedger` orchestrates the simulation/backtest loop and provides the main hedging workflows.

What it does:
- Computes option value and Greeks over time (`greeks`)
- Runs hedging experiments:
  - **Delta hedging** (`run_delta_hedge`)
  - **Delta–Gamma hedging** (`run_delta_gamma_hedge`)
- Provides diagnostics/plots:
  - hedging performance and PnL (`plot_results`)
  - gamma/theta-related metrics (`plot_gamma_theta_metrics`)
- Includes Monte Carlo utilities:
  - GBM path simulation (`simulate_gbm_paths`)
  - uncertainty analysis (`run_uncertainty_mc`, `plot_uncertainty`)

### 3) Scenario extensions
The notebook then extends the base engine with focused “what-if” modules, following the heritage principle of the OOP. 

#### `OptionHedgerWithCrash`
Adds **crash-style paths and volatility/smile shifts**:
- Path generators: `make_crash_path`, `make_vol_shift_path`
- Smile shift mechanics: `_shift_smile_equations`
- A dedicated workflow to hedge under smile shifts: `run_delta_hedge_with_smile_shift`
- Theta-focused crash analysis: `analyze_theta_hedging_in_crash`
- Visualization helpers: `_plot_crash_theta`, `_plot_dvol`

#### `OptionHedgerWithRates`
Introduces **time-varying interest rates** and **rho-aware hedging**:
- Rate scenario generator: `make_rate_path`
- Delta–Rho hedging: `run_delta_rho_hedge`
- Interpretation plots: `plot_rho_role`

### 4) Transaction costs & execution microstructure
A small microstructure layer models trading costs and execution.

- `FeeModel` (interface): `fee(notional)`
- `FixedBpsFee`: proportional fees in basis points
- `MinFloorFee`: proportional fees with a minimum floor
- `Broker`: wraps a `FeeModel` and computes commissions (`commission`)
- `ExecutionModel`: defines execution price logic (`execute_price`)

`OptionHedgerWithCosts` ties these components into the hedging loop:
- runs hedging with explicit costs: `run_hedge_with_costs`
- hedging Greeks for the hedge instrument: `hedge_greeks`
- impact reporting: `plot_cost_impact`

### 5) Dedicated Delta–Gamma engine: `OptionHedgerDG`
`OptionHedgerDG` is a specialized delta–gamma hedger that explicitly handles a **gamma-hedge option**:
- `hedge_greeks` computes the Greeks/value of the gamma hedge instrument (single option or multi-leg strategy)
- delta–gamma hedge workflow: `run_delta_gamma_hedge`
- MC uncertainty analysis for delta–gamma setups: `run_uncertainty_mc_dg`, `plot_uncertainty`


## Econometric Analysis

The notebook `financial_derivatives_econometrics.ipynb` focuses on **econometric and statistical analysis**, such as exploratory data analysis, regression models, or time-series methods, depending on the studied variables.

These analyses are intended to complement the hedging results by providing a deeper understanding of market dynamics and model behavior.

## Suggested Workflow

1. Run `backtester.ipynb` from start to finish.
2. Adjust scenario parameters (rebalancing frequency, volatility assumptions, costs).
3. Analyze results and PnL decomposition.
4. Explore `financial_derivatives_econometrics.ipynb` for statistical validation and insights.
5. Optionally explore or extend `ml_prediction.py`.

## Notes and Limitations

* Results are highly sensitive to modeling assumptions (volatility, transaction costs, discretization).
* This project is primarily **educational and experimental**, not intended for direct trading use.
* For production-level use, improvements could include:

  * Unit testing
  * Configuration files (YAML/JSON)
  * Clear separation between library code and notebooks

## Author

* GitHub: **Thomasdeport**
