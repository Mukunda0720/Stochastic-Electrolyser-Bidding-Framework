# electrolyzer_definitions.py
# V5.0 - Added storage, compressor, and min-on-level for the electrolyzer.

import numpy as np
import pandas as pd

# Target for final product (e.g., in kg)
# Based on 55MW capacity * 18 kg/MWh efficiency * 24 hours
TARGET_HYDROGEN_PER_DAY = 23760.0 # kg

# --- CHANGE: Re-introduce STORAGE_DEFINITIONS ---
# We now have a storage tank for the hydrogen produced by the electrolyzer.
STORAGE_DEFINITIONS = {
    'Hydrogen': {'Min_Level': 0.0, 'Max_Level': 10000.0, 'Initial_Level': 5000.0}, # Max 50,000 kg, start with 10,000 kg
}

# --- CHANGE: Update PLANT_DEFINITIONS to include the Compressor ---
PLANT_DEFINITIONS = {
    'Electrolyzer': {
        'Primary_Input': 'Electricity_for_H2',
        'Outputs': {'Hydrogen': 18.0}, # Produces raw Hydrogen into storage
        'Other_Inputs': {},
        'Average_Capacity': 55.0, # MW
        'Operational_Min': 0.0,
        # --- CHANGE: Add a minimum operating level when the unit is on ---
        'Min_Op_When_On': 0.10, # Must operate at >=10% of capacity if on
        'Operational_Max': 1.0,
        'Ramp_Up_Rate': 1.0,
        'Ramp_Down_Rate': 1.0,
        'Is_Fixed': False
    },
    # --- CHANGE: Add the new Compressor plant ---
    'Compressor': {
        'Primary_Input': 'Hydrogen', # Takes Hydrogen from storage
        'Outputs': {'Compressed_Hydrogen': 1.0}, # Final, sellable product
        'Other_Inputs': {
            # Consumes electricity for compression, e.g., 2 kWh per kg H2 (0.002 MWh/kg)
            'Electricity': -0.002
        },
        'Average_Capacity': 1500.0, # Can compress 1500 kg of H2 per hour
        'Operational_Min': 0.0,
        'Operational_Max': 1.0,
        'Ramp_Up_Rate': 1.0, # Can turn on/off instantly
        'Ramp_Down_Rate': 1.0,
        'Is_Fixed': False
    }
}


# --- The rest of the file populates globals based on the definitions above ---
# This part of the script does not need manual changes; it will adapt automatically.

MATERIALS_IN_STORAGE = list(STORAGE_DEFINITIONS.keys())
FLEXIBLE_PLANTS = [name for name, params in PLANT_DEFINITIONS.items() if not params.get('Is_Fixed', False)]
FIXED_PLANTS = []

ABSOLUTE_PLANT_LIMITS = {}
ELECTRICITY_INPUT_MAP = {}

def calculate_plant_parameters():
    global ABSOLUTE_PLANT_LIMITS, ELECTRICITY_INPUT_MAP
    for name, params in PLANT_DEFINITIONS.items():
        avg_cap_val = params.get('Average_Capacity')
        op_min_rate = params.get('Operational_Min', 0.0)
        op_max_rate = params.get('Operational_Max', 1.0)
        ramp_up_rate_val = params.get('Ramp_Up_Rate', 1.0)
        ramp_down_rate_val = params.get('Ramp_Down_Rate', 1.0)
        # --- CHANGE: Get the new Min_Op_When_On parameter ---
        min_op_when_on_rate = params.get('Min_Op_When_On', 0.0)

        if avg_cap_val is None or pd.isna(avg_cap_val):
            ABSOLUTE_PLANT_LIMITS[name] = {
                'Min_Op_Abs': np.nan, 'Max_Op_Abs': np.nan,
                'Ramp_Up_Abs': np.nan, 'Ramp_Down_Abs': np.nan,
                'Min_Op_When_On_Abs': np.nan
            }
        else:
            ABSOLUTE_PLANT_LIMITS[name] = {
                'Min_Op_Abs': avg_cap_val * op_min_rate,
                'Max_Op_Abs': avg_cap_val * op_max_rate,
                'Ramp_Up_Abs': avg_cap_val * ramp_up_rate_val,
                'Ramp_Down_Abs': avg_cap_val * ramp_down_rate_val,
                # --- CHANGE: Calculate the absolute value for the min-on level ---
                'Min_Op_When_On_Abs': avg_cap_val * min_op_when_on_rate
            }

        # For the Electrolyzer, the operating level is MW, so the electricity factor is 1.
        # For the Compressor, the operating level is kg/hr, so we need the electricity factor.
        if name == 'Electrolyzer':
            ELECTRICITY_INPUT_MAP[name] = 1.0
        else:
            elec_eff = params.get('Other_Inputs', {}).get('Electricity', 0.0)
            ELECTRICITY_INPUT_MAP[name] = elec_eff


# Automatically run the calculation when the module is imported
calculate_plant_parameters()
