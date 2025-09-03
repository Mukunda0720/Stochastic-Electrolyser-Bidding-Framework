# Electrolyzer Bidding Strategy Optimisation Framework

## 1. Overview

Driven by the energy transition and the rise of intermittent renewables, electricity price volatility presents a significant challenge for large industrial consumers. This project, the implementation of a Master's thesis from TU Delft, provides a simulation and optimisation framework to address this challenge.

The framework determines the optimal hourly bidding strategy for the day-ahead electricity market, specifying the price and quantity of electricity to procure for each hour of the following day. It uses a stochastic rolling-horizon optimisation approach, leveraging machine learning for electricity price forecasting and scenario generation to account for market uncertainty. The primary objective is to maximise the profitability of the electrolyser plant by strategically creating hourly bid curves, while meeting specified hydrogen production targets and respecting the physical and operational constraints of the equipment.

---

## 2. Academic Context & Citation

This codebase is the implementation for the Master of Science thesis:

**Title:** Optimising Industrial Participation in the Day-Ahead Electricity Market: A  Stochastic Bidding Framework with Risk Management  
**Author:** Mukunda Badarinath  
**Institution:** Delft University of Technology  
**Programme:** Master of Science in Sustainable Energy Technology

For a detailed explanation of the theoretical background, mathematical model formulation, risk management approach (CVaR), and case study results, please refer to the full thesis document (`Mukunda_Master_Thesis.pdf`).

---

## 3. System Architecture & Workflow

The simulation operates in a rolling-horizon loop. For each decision window (e.g., a 5-day planning horizon), the following sequence of operations is performed:

1. **Data Ingestion:** Historical market data, including prices and forecasts for renewables and load, is loaded from an input file.
2. **ML Price Forecasting:** A deterministic price forecast for the upcoming planning horizon is generated using a Prophet time series model.
3. **Stochastic Scenario Generation:** Based on the forecast and historical volatility, a large number of potential price scenarios are generated. These scenarios are then reduced to a computationally tractable set using k-means clustering.
4. **Stochastic Optimisation:** The core optimisation model, built in Pyomo, is solved. It co-optimises the hourly bid curve (price and quantity pair for each hour) and the operational schedule across all scenarios to maximise expected profit while managing risk using Conditional Value-at-Risk (CVaR).
5. **Bid Submission:** The optimal hourly bids for the first implementation period (i.e. the next 24 hours) are extracted from the optimisation result.
6. **Market Clearing Simulation:** The submitted bids are cleared against the actual (historical) market prices for that period to determine the accepted electricity volume and the resulting cost.
7. **State Update & Roll Forward:** The simulation clock advances, the electrolyser and storage states are updated based on the cleared schedule, and the process repeats for the next decision window.
8. **Results Aggregation:** All operational and financial results are collected and saved to output files.

---

## 4. Model Formulation & Thesis Reference

The core optimisation model is formulated as a stochastic mixed-integer linear programme (MILP) in `stochastic_electrolyzer_bidding.py`. For a detailed explanation of the mathematical formulation, objective function, and all system constraints (including bidding, ramping, storage, and risk management), please refer to the full thesis document.

---

## 5. File Descriptions & Code Flow

Below are the key files in the order they are used during a typical simulation run:

- **`run_electrolyzer_simulation.py`**: Main entry point. Handles command-line arguments and initiates the main simulation manager.
- **`config.yaml`**: Central configuration file. Loaded at start-up, provides all necessary parameters for the simulation.
- **`electrolyzer_rolling_horizon.py`**: Simulation orchestrator. Coordinates the entire process, calling forecasting, scenario generation, optimisation, and market clearing modules in sequence.
- **`scaled_data_NL4.xlsx - Sheet1.csv`**: Input data. Contains historical time-series data (prices, load renewable energy generation forecast and total generation forecast) used as the basis for back-testing the bidding strategy and also has the training set for the Prophet time series model.
- **`ml_forecaster.py`**: Price forecaster. Uses historical data to train a Prophet model and generate a deterministic price forecast.
- **`ml_scenario_generator.py`**: Uncertainty modeller. Generates a large set of stochastic price scenarios and reduces them using k-means clustering.
- **`electrolyzer_definitions.py`**: Physical asset definitions. Defines technical parameters of the physical assets (capacity, efficiency, ramp rates, storage limits).
- **`stochastic_electrolyzer_bidding.py`**: Core optimisation engine. Builds and solves the large-scale stochastic MILP model in Pyomo.
- **`simple_market_clearing.py`**: Market simulator. Simulates the day-ahead market clearing process.
- **`synthetic_prices.py`**: Utility data generator. Can be enabled in `config.yaml` to generate synthetic price data for testing.

---

## 6. Prerequisites & Installation

### A. Python Libraries

You will need Python 3.8 or newer. Install the required libraries via pip:

```sh
pip install pandas numpy pyomo prophet scikit-learn pyyaml matplotlib scipy
```

### B. Optimisation Solver

This framework uses Pyomo, which requires a separate solver. The default is Gurobi (commercial):

- **Install Gurobi:** Download and install from the [Gurobi website](https://www.gurobi.com/). You may need an academic or commercial licence.
- **Install Gurobi Python Bindings:**  
  ```sh
  pip install gurobipy
  ```
- **Ensure Pyomo can find the solver:** Make sure the Gurobi executable is in your system's PATH.

Alternatively, you can modify the `solver: name:` parameter in `config.yaml` to use another supported solver (e.g., glpk, cbc, cplex), but performance may vary.

---

## 7. How to Run the Simulation

1. **Prepare Input Data:**  
   Ensure your historical market data is in a CSV file and the path is correctly specified in `config.yaml` under `simulation: market_input_file`. The file must contain a `Date` column, `Price` column, `Solar_Wind_Forecast` column, `Load_Forecast` column and `Total_Generation_Forecast`. These data are provided for the period of 2022-2024. New hourly data can be accessed on https://newtransparency.entsoe.eu/.

2. **Configure the Simulation:**  
   Open `config.yaml` and adjust parameters to define your simulation case. Key parameters include:
   - `simulation`: Set `start_date`, `end_date`, `planning_horizon`, and `output_dir`.
   - `electrolyzer_plant`: Define `target_hydrogen_per_day`.
   - `risk_parameters`: Adjust `cvar_alpha` (confidence level) and `cvar_lambda_weight` (risk aversion).
   - `solver`: Configure solver options like `TimeLimit`.

3. **Execute the Run Script:**  
   Open your terminal, navigate to the project directory, and run:

   ```sh
   python run_electrolyzer_simulation.py --config config.yaml
   ```

   You can omit the `--config` argument if your configuration file is named `config.yaml` and is in the same directory.

4. **Analyse Results:**  
   The simulation will print progress to the console. Upon completion, all results, including economic summaries, operational schedules, and bids, will be saved as `.xlsx` files in the directory specified by `output_dir` in the config file.

---

## 8. Configuration Details (`config.yaml`)

This file is central to controlling the model's behaviour. Key sections:

- **simulation:** Defines the overall simulation timeline, planning window length, and input/output files.
- **electrolyzer_plant:** Sets the daily hydrogen production target.
- **objective_tuning:** Allows enabling penalties for production shortfalls (not used in the primary model version).
- **risk_parameters:** Controls the Conditional Value-at-Risk (CVaR) formulation.
- **ml_forecasting & scenario_generation:** Parameters for the Prophet model and scenario generation logic.
- **bid_strategy:** Constraints on the bids, like the maximum number of distinct price/quantity blocks per hour.
- **solver:** Specifies the solver name and options.
- **synthetic_data:** Allows you to bypass the ML forecaster and use the synthetic price generator by setting `enabled: true`. This allows you to define your own market price behaviour and also the accuracy of the forecast. This can be used for stress testing.

---

## 9. Author

Mukunda