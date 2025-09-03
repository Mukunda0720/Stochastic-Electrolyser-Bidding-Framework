# utils_ramp.py
"""
Utility functions for ramp calculations related to electrolyzer optimization.
"""
import pandas as pd # Required for pd.isna check if ABSOLUTE_PLANT_LIMITS might contain NaN from definitions
from electrolyzer_definitions import (
    FLEXIBLE_PLANTS,
    ABSOLUTE_PLANT_LIMITS,
    ELECTRICITY_INPUT_MAP,
)

def compute_ramp_envelopes_mw() -> tuple[float, float]:
    """
    Calculates the fleet-level maximum electricity ramp-up and ramp-down
    capability in MW for a 1-hour step based on one-way ramp limits.

    Returns:
        tuple[float, float]: (delta_E_up_max_mw, delta_E_down_max_mw)
                             delta_E_up_max_mw: Max MW that can be added by the fleet in 1 hour.
                             delta_E_down_max_mw: Max MW that can be removed by the fleet in 1 hour.
    """
    env_up_mw = 0.0   # Total MW that can be added by the fleet
    env_dn_mw = 0.0   # Total MW that can be removed by the fleet

    for plant_name in FLEXIBLE_PLANTS:
        electricity_coeff = abs(ELECTRICITY_INPUT_MAP.get(plant_name, 0.0)) # MWh/ton or MWh/MWh

        if electricity_coeff < 1e-9: # Effectively zero
            continue

        plant_limits = ABSOLUTE_PLANT_LIMITS.get(plant_name)
        if not plant_limits:
            continue

        ramp_up_abs_plant = plant_limits.get("Ramp_Up_Abs")
        ramp_down_abs_plant = plant_limits.get("Ramp_Down_Abs")

        if ramp_up_abs_plant is None or pd.isna(ramp_up_abs_plant):
            ramp_up_abs_plant = 0.0
        if ramp_down_abs_plant is None or pd.isna(ramp_down_abs_plant):
            ramp_down_abs_plant = 0.0
            
        env_up_mw += ramp_up_abs_plant * electricity_coeff
        env_dn_mw += ramp_down_abs_plant * electricity_coeff
        
    return env_up_mw, env_dn_mw

def compute_fleet_symmetric_ramp_capability_mw() -> float:
    """
    Calculates the fleet-level maximum electricity symmetric ramp capability in MW.
    This is the maximum amount the fleet can ramp up by in one hour and then
    immediately ramp down by (or vice-versa) in the next hour, returning to
    the initial state. It's limited by the sum of min(individual plant ramp-up,
    individual plant ramp-down) capabilities.

    Returns:
        float: fleet_symmetric_ramp_mw
    """
    fleet_symmetric_ramp_mw = 0.0

    for plant_name in FLEXIBLE_PLANTS:
        electricity_coeff = abs(ELECTRICITY_INPUT_MAP.get(plant_name, 0.0)) # MWh/ton or MWh/MWh

        if electricity_coeff < 1e-9: # Effectively zero
            # print(f"Debug Symmetric: Plant {plant_name} has zero electricity coefficient. Skipping.")
            continue

        plant_limits = ABSOLUTE_PLANT_LIMITS.get(plant_name)
        if not plant_limits:
            # print(f"Debug Symmetric: Plant {plant_name} not found in ABSOLUTE_PLANT_LIMITS. Skipping.")
            continue

        # Get physical ramp rates (tons/hr or MW for Linde)
        physical_ramp_up_abs_plant = plant_limits.get("Ramp_Up_Abs")
        physical_ramp_down_abs_plant = plant_limits.get("Ramp_Down_Abs")

        # Handle potential NaN or None values
        if physical_ramp_up_abs_plant is None or pd.isna(physical_ramp_up_abs_plant):
            # print(f"Debug Symmetric: Ramp_Up_Abs for plant {plant_name} is missing or NaN. Treating as 0.")
            physical_ramp_up_abs_plant = 0.0
        if physical_ramp_down_abs_plant is None or pd.isna(physical_ramp_down_abs_plant):
            # print(f"Debug Symmetric: Ramp_Down_Abs for plant {plant_name} is missing or NaN. Treating as 0.")
            physical_ramp_down_abs_plant = 0.0
        
        # The plant's symmetric physical ramp capability is the minimum of its up and down ramp rates
        symmetric_physical_ramp_plant = min(physical_ramp_up_abs_plant, physical_ramp_down_abs_plant)
        
        # Convert this to MW
        symmetric_mw_ramp_plant = symmetric_physical_ramp_plant * electricity_coeff
        fleet_symmetric_ramp_mw += symmetric_mw_ramp_plant
        
        # print(f"Debug Symmetric: Plant {plant_name}: PhysRampUp={physical_ramp_up_abs_plant:.2f}, PhysRampDn={physical_ramp_down_abs_plant:.2f}, MinPhysRamp={symmetric_physical_ramp_plant:.2f}, Coeff={electricity_coeff:.4f}, SymmMWRamp={symmetric_mw_ramp_plant:.2f}")

    # print(f"Computed Fleet Symmetric Ramp Capability: {fleet_symmetric_ramp_mw:.2f} MW")
    return fleet_symmetric_ramp_mw


if __name__ == '__main__':
    print("Attempting to compute ramp envelopes...")
    try:
        up_env, dn_env = compute_ramp_envelopes_mw()
        print(f"Fleet-wide Max One-Way Ramp-Up Capability (ΔE↑_max): {up_env:.2f} MW per hour")
        print(f"Fleet-wide Max One-Way Ramp-Down Capability (ΔE↓_max): {dn_env:.2f} MW per hour")

        symmetric_env = compute_fleet_symmetric_ramp_capability_mw()
        print(f"Fleet-wide Max Symmetric Ramp Capability: {symmetric_env:.2f} MW per hour")

    except ImportError as e:
        print(f"ImportError: {e}. Make sure electrolyzer_definitions.py is accessible.")
    except KeyError as e:
        print(f"KeyError: {e}. This might indicate that ABSOLUTE_PLANT_LIMITS or ELECTRICITY_INPUT_MAP is not populated correctly.")
        print("Ensure 'calculate_plant_parameters()' in 'electrolyzer_definitions.py' has run and populated these globals.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
