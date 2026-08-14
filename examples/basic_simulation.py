"""Run a small deterministic thermal simulation."""

import numpy as np

from thermal_digital_twin import ThermalParameters, simulate_temperature


time_s = np.linspace(0, 600, 121)
ambient_c = np.full_like(time_s, 25.0)
power_w = np.full_like(time_s, 180.0)

temperature_c = simulate_temperature(
    time_s,
    ambient_c,
    power_w,
    initial_temperature_c=25.0,
    params=ThermalParameters(ua=12.0, heat_capacity=2500.0),
)

print(f"Final temperature: {temperature_c[-1]:.2f} °C")
