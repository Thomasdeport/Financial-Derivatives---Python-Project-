# Financial Derivatives — Python Project

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
