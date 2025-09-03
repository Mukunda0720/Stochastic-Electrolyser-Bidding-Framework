# Price-Insensitive Bidding Strategy Framework

## 1. Overview

This project provides a simulation framework for a **Price-Insensitive bidding strategy** ("Plan First, Buy Later") for grid-connected hydrogen production. It serves as a baseline in the TU Delft Master's thesis, representing a traditional industrial procurement model: production is planned based on physical needs to meet the production target, and electricity is purchased from the market by bidding at the market cap prices.

**Core Logic:**
- **Plan First:** A deterministic optimisation model creates the most efficient 24-hour operational schedule to meet a specified hydrogen production target, ignoring electricity prices and focusing solely on physical constraints (ramp rates, capacity, storage).
- **Buy Later:** The resulting hourly electricity consumption profile is converted into bids for the day-ahead market. Each hourly consumption value is submitted as a single bid at a very high price (market price cap), guaranteeing acceptance.

This strategy is used as a benchmark to evaluate the economic benefits of more advanced, price-responsive bidding models.

---

## 2. Academic Context & Citation

This codebase is the implementation for the Master of Science thesis:

**Title:** Optimising Industrial Participation in the Day-Ahead Electricity Market: A  Stochastic Bidding Framework with Risk Management  
**Author:** Mukunda Badarinath  
**Institution:** Delft University of Technology  
**Programme:** Master of Science in Sustainable Energy Technology


The Price-Insensitive model serves as the benchmark against which the "Hourly Bids" and "Exclusive Group Bids" strategies are compared. For full details, please refer to the thesis document.

---

## 3. System Architecture & Workflow

The simulation operates in a simplified rolling-horizon loop **without price forecasting or uncertainty modelling**:

1. **Data Ingestion:** Historical market data is loaded, but only used after the plan is made, during market clearing.
2. **Deterministic Planning:** The core optimisation model creates an optimal physical production plan for the upcoming horizon, meeting the hydrogen production target at the lowest possible physical effort, without considering electricity prices.
3. **Bid Formulation:** The hourly electricity consumption schedule is converted into a series of price-insensitive bids (each hour's required electricity is submitted as a single block at the maximum market price).
4. **Market Clearing Simulation:** The submitted high-price bids are cleared against actual (historical) market prices. Since the bid prices are at the cap, all required electricity is "accepted," and the cost is calculated based on actual market prices.
5. **State Update & Roll Forward:** The simulation clock advances, asset states are updated, and the process repeats.
6. **Results Aggregation:** All financial and operational results are collected and saved to output files.

---

## 4. File Descriptions & Code Flow

Key files (in order of use during a typical simulation run):

- **[`run_electrolyzer_simulation_plan_first.py`](run_electrolyzer_simulation_plan_first.py):** Main entry point. Execute this script to start the simulation for the Price-Insensitive strategy.
- **[`config_plan_first.yaml`](config_plan_first.yaml):** Central configuration file. Provides all necessary parameters, especially `target_hydrogen_per_day`.
- **[`electrolyzer_rolling_horizon_plan_first.py`](electrolyzer_rolling_horizon_plan_first.py):** Simulation orchestrator. Coordinates the workflow, calling the deterministic planner and market clearing module.
- **[`deterministic_electrolyzer_planner.py`](deterministic_electrolyzer_planner.py):** Core optimisation engine. Contains the Pyomo model that solves for the physically optimal production plan (ignores electricity prices).
- **[`electrolyzer_definitions.py`](electrolyzer_definitions.py):** Physical asset definitions. Defines technical parameters (capacity, efficiency, ramp rates) used as constraints in the planner.
- **[`simple_market_clearing.py`](simple_market_clearing.py):** Market simulator. Simulates market clearing against actual historical prices to determine the final electricity cost.
- **[`synthetic_prices.py`](synthetic_prices.py):** Utility for generating synthetic price data (optional, for testing).
- **[`plot_electrolyzer_plan_first_results.py`](plot_electrolyzer_plan_first_results.py):** Script for visualising simulation results.
- **[`utils_ramp.py`](utils_ramp.py):** Utility functions for ramping constraints and solver diagnostics.
- **`scaled_data_NL4.xlsx`:** Example input data file (historical market data).

---

## 5. Prerequisites & Installation

### A. Python Libraries

You will need Python 3.8 or newer. Install the required libraries via pip:

```sh
pip install pandas numpy pyomo pyyaml matplotlib openpyxl
```

### B. Optimisation Solver

This framework uses Pyomo, which requires a separate solver. The configuration is set to use Gurobi.

1. **Install Gurobi:** Download and install the Gurobi solver from their official website.
2. **Install Gurobi Python Bindings:** `pip install gurobipy`
3. **Ensure Pyomo can find the solver:** Make sure the Gurobi executable is in your system's PATH.

---

## 6. How to Run the Simulation

1. **Configure the Simulation:** Open `config_plan_first.yaml` and adjust the parameters. The most critical parameter is `electrolyzer_plant: target_hydrogen_per_day`.
2. **Execute the Run Script:** Open your terminal, navigate to the project directory, and run the main script for this strategy:

   ```sh
   python run_electrolyzer_simulation_plan_first.py --config config_plan_first.yaml
   ```

   > **Note:** The `--config` argument is optional if the file is named `config_plan_first.yaml` and is in the same directory.

3. **Analyse Results:** The simulation will print progress to the console. All results (economic summaries, operational schedules, etc.) will be saved as .xlsx files in the directory specified by `output_dir` in the config file.

---

## 7. Author

Mukunda Badarinath