# Exclusive Group Bidding Strategy Optimisation Framework

## 1. Overview

This project provides a comprehensive simulation and optimisation framework to determine the optimal day-ahead electricity market bidding strategy for a grid-connected green hydrogen production facility. It is the implementation of the "Exclusive Group Bids" methodology described in the Master's thesis from TU Delft.

Unlike granular hourly bidding, the Exclusive Group Bidding strategy operates on a set of pre-defined, mutually exclusive daily operational profiles. For example, these profiles could represent "Maximum Production," "Night-Only Production," or "Risk averse" modes. The framework's core task is to select the single, most profitable operational portfolio of profiles for the entire upcoming day, given the uncertainty of market prices.

The framework uses a stochastic optimisation approach, leveraging machine learning for price forecasting and scenario generation to select the optimal daily profile under uncertainty.

---

## 2. Academic Context & Citation

This codebase is the implementation for the Master of Science thesis:

**Title:** Optimising Industrial Participation in the Day-Ahead Electricity Market: An Adaptive Stochastic Bidding Framework with Risk Management  
**Author:** Mukunda Badarinath  
**Institution:** Delft University of Technology  
**Programme:** Master of Science in Sustainable Energy Technology

For a detailed explanation of the theoretical background, the distinction between Hourly and Exclusive Group bids, and case study results, please refer to the full thesis document.

---

## 3. System Architecture & Workflow

The simulation operates in a rolling-horizon loop. For each decision window, the following sequence is performed:

1. **Data Ingestion:** Historical market data is loaded from an input file.
2. **ML Price Forecasting:** A deterministic price forecast for the upcoming planning horizon is generated using a Prophet model.
3. **Stochastic Scenario Generation:** A large number of potential price scenarios are generated to represent market uncertainty and are then reduced to a tractable set.
4. **Stochastic Optimisation:** The core optimisation model evaluates a set of pre-defined, mutually exclusive operational profiles against all price scenarios. It calculates the expected profit and risk (CVaR) for each potential profile.
5. **Profile Selection & Bid Submission:** The single operational profile that offers the highest risk-adjusted expected profit is selected. This is determined by maximising the average profit across all scenarios while simultaneously minimising the Conditional Value-at-Risk (CVaR) to manage downside financial risk. This profile's 24-hour consumption pattern represents the bid for the next day.
6. **Market Clearing Simulation:** The chosen profile is cleared against the actual (historical) market prices to determine the electricity cost and final operational schedule.
7. **State Update & Roll Forward:** The simulation clock advances, asset states are updated based on the cleared schedule, and the process repeats.
8. **Results Aggregation:** All financial and operational results are saved to output files.

---

## 4. File Descriptions & Code Flow

Below are the key files in the order they are used during a typical simulation run:

- **`run_electrolyzer_simulation_eg.py`**: Main entry point. This is the script you execute to start the simulation. It specifically calls the manager for the Exclusive Group Bids model.
- **`config_eg.yaml`**: Central configuration file. This YAML file provides all necessary parameters for the simulation, including dates, file paths, risk parameters, and solver settings.
- **`electrolyzer_rolling_horizon_eg.py`**: Simulation orchestrator. Its main loop coordinates the entire process, calling the forecasting, scenario generation, optimisation, and market clearing modules in sequence.
- **`ml_forecaster.py` & `ml_scenario_generator.py`**: Uncertainty modelling. These modules are responsible for generating the deterministic price forecast (Prophet) and the subsequent stochastic price scenarios (K-Means clustering) that feed into the optimiser.
- **`electrolyzer_definitions.py`**: Physical asset definitions. This file defines the technical parameters of the physical assets (electrolyzer capacity, efficiency, ramp rates, storage limits), which are used to construct the operational constraints.
- **`stochastic_electrolyzer_bidding_eg.py`**: Core optimisation engine. This is the new optimisation heart. It builds and solves a stochastic mixed-integer linear programme in Pyomo. Its core logic is to select the single best daily operational profile ("exclusive group"). The "best" profile is defined as the one that maximises the risk-adjusted expected profit, which is a balance between achieving the highest average profit across all price scenarios and minimising the financial risk in worst-case scenarios (managed via Conditional Value-at-Risk, CVaR).
- **`simple_market_clearing_eg.py`**: Market simulator. After the optimiser selects the best daily profile (the one with the highest risk-adjusted profit), this module simulates the market clearing process. It compares the chosen profile's consumption against actual historical prices to determine the final electricity cost.

---

## 5. Prerequisites & Installation

### A. Python Libraries

You will need Python 3.8 or newer. The required libraries can be installed via pip:

```sh
pip install pandas numpy pyomo prophet scikit-learn pyyaml matplotlib joblib openpyxl scipy
```

### B. Optimisation Solver

This framework uses Pyomo, which requires a separate solver. The configuration is set to use Gurobi.

- **Install Gurobi:** Download and install the Gurobi solver from their [official website](https://www.gurobi.com/).
- **Install Gurobi Python Bindings:**  
  ```sh
  pip install gurobipy
  ```
- **Ensure Pyomo can find the solver:** Make sure the Gurobi executable is in your system's PATH.

---

## 6. How to Run the Simulation

1. **Prepare Input Data:**  
   Ensure your historical market data is in a CSV file and the path is correctly specified in `config.yaml` under `simulation: market_input_file`. The file must contain a `Date` column, `Price` column, `Solar_Wind_Forecast` column, `Load_Forecast` column and `Total_Generation_Forecast`. These data are provided for the period of 2022-2024. New hourly data can be accessed on https://newtransparency.entsoe.eu/.

2. **Configure the Simulation:**  
   Open `config_eg.yaml` and adjust parameters to define your simulation case. Key parameters include:
   - `simulation`: Set `start_date`, `end_date`, `planning_horizon`, and `output_dir`.
   - `electrolyzer_plant`: Define `target_hydrogen_per_day`.
   - `risk_parameters`: Adjust `cvar_alpha` (confidence level) and `cvar_lambda_weight` (risk aversion).
   - `solver`: Configure solver options like `TimeLimit`.

3. **Execute the Run Script:**  
   Open your terminal, navigate to the project directory, and run the main script for the exclusive group model:

   ```sh
   python run_electrolyzer_simulation_eg.py --config config_eg.yaml
   ```

   *Note: The `--config` argument is optional if the file is named `config_eg.yaml` and is in the same directory.*

4. **Analyse Results:**  
   The simulation will print progress to the console. All results (economic summaries, operational schedules, etc.) will be saved as `.xlsx` files in the directory specified by `output_dir` in the config file.

---

## 8. Configuration Details (`config_eg.yaml`)

This file is central to controlling the model's behaviour. Key sections:

- **simulation:** Defines the overall simulation timeline, planning window length, and input/output files.
- **electrolyzer_plant:** Sets the daily hydrogen production target.
- **objective_tuning:** Allows enabling penalties for production shortfalls (not used in the primary model version).
- **risk_parameters:** Controls the Conditional Value-at-Risk (CVaR) formulation.
- **ml_forecasting & scenario_generation:** Parameters for the Prophet model and scenario generation logic.
- **bid_strategy:** Constraints on the bids, like the maximum number of profiles that can be made and the factors with which the ranked profiles will be priced at with respect to the set opportunity cost.
- **solver:** Specifies the solver name and options.
- **synthetic_data:** Allows you to bypass the ML forecaster and use the synthetic price generator by setting `enabled: true`. This allows you to define your own market price behaviour and also the accuracy of the forecast. This can be used for stress testing.

--


## 8. Author

Mukunda Badarinath