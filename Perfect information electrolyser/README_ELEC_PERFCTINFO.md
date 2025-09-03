# Perfect Information Bidding Strategy Framework

## 1. Overview

This project provides a simulation framework for a **Perfect Information** bidding strategy, also known as a "Perfect Foresight" or "Oracle" model. This is a theoretical, non-realizable strategy that serves as the ultimate benchmark in the Master's thesis from TU Delft.

The purpose of this model is to calculate the maximum possible economic profit an operator could achieve if they had perfect knowledge of the actual day-ahead market prices for the entire simulation period. It achieves this by solving a single, large-scale optimisation problem that covers the entire duration (e.g., a full month) in one run, using the real historical prices as direct inputs.

> **Note:** Because it operates with perfect knowledge, this strategy is not practical for real-world bidding. Instead, its results represent the theoretical "upper bound" of performance, providing a vital benchmark against which the effectiveness of more realistic, forecast-based strategies (like Hourly Bids and Exclusive Group Bids) can be measured.

---

## 2. Academic Context & Citation

This codebase is the implementation for the Master of Science thesis:

**Title:** Optimising Industrial Participation in the Day-Ahead Electricity Market: A  Stochastic Bidding Framework with Risk Management  
**Author:** Mukunda Badarinath  
**Institution:** Delft University of Technology  
**Programme:** Master of Science in Sustainable Energy Technology


This Perfect Information model establishes the theoretical maximum performance, serving as the key benchmark for evaluating all other bidding strategies discussed in the thesis.

---

## 3. System Architecture & Workflow

The simulation for this strategy is **not** a rolling-horizon process. It is a single, comprehensive optimisation run:

1. **Data Ingestion:**  
   The actual, historical market prices for the entire simulation period (e.g., one month) are loaded into memory at the start.

2. **Global Deterministic Optimisation:**  
   A single, large optimisation model is constructed for the entire period. The model is given the actual historical prices and is tasked with creating a globally optimal operational schedule that maximises profit (revenue from hydrogen sales minus electricity costs) while respecting all physical asset constraints.

3. **Result Processing:**  
   The solver finds the single best operational plan. Since the plan was created using the real prices, no separate "market clearing" simulation is needed. The results from the optimisation are the final results.

4. **Results Aggregation:**  
   The complete, globally optimal financial and operational results are saved to output files.

---

## 4. File Descriptions & Code Flow

This section describes the key files in the order they are used:

- **`run_perfect_information_simulation.py`**  
  Main entry point. This is the script you execute. It manages the entire process: loading all the data for the full period, calling the planner once, and saving the final results.

- **`config_perfect_info.yaml`**  
  Central configuration file. This file defines the simulation period (`start_date`, `end_date`), file paths, and key economic parameters like the `hydrogen_price` used in the objective function.

- **`perfect_information_planner.py`**  
  Core optimisation engine. This is the heart of the model. It builds and solves a single, large-scale Pyomo model that covers the entire simulation duration. It takes the actual historical prices as a direct input to find the most profitable schedule possible.

- **`electrolyzer_definitions.py`**  
  Physical asset definitions. This shared file defines the technical parameters of the electrolyser (capacity, efficiency, ramp rates), which are used as constraints in the optimisation model.

- **`simple_market_clearing.py`**  
  This file is included for consistency but is not used in the perfect information workflow, as the planner's output is already the final, cleared result.

---

## 5. Prerequisites & Installation

### A. Python Libraries

You will need Python 3.8 or newer. The required libraries can be installed via pip:

```sh
pip install pandas numpy pyomo pyyaml matplotlib openpyxl
```

### B. Optimisation Solver

This framework uses Pyomo, which requires a separate solver. The configuration is set to use **Gurobi**.

- **Install Gurobi:** Download and install the Gurobi solver from their [official website](https://www.gurobi.com/).
- **Install Gurobi Python Bindings:**  
  ```sh
  pip install gurobipy
  ```
- **Ensure Pyomo can find the solver:**  
  Make sure the Gurobi executable is in your system's PATH.

---

## 6. How to Run the Simulation

1. **Configure the Simulation:**  
   Open `config_perfect_info.yaml` and set the `start_date`, `end_date`, and `output_dir`.

2. **Execute the Run Script:**  
   Open your terminal, navigate to the project directory, and run:

   ```sh
   python run_perfect_information_simulation.py --config config_perfect_info.yaml
   ```

   > The `--config` argument is optional if the file is named `config_perfect_info.yaml` and is in the same directory.

3. **Analyse Results:**  
   The simulation will print progress to the console. The final, globally optimal results will be saved as `.xlsx` files in the directory specified by `output_dir`.

---

## 7. Author

Mukunda