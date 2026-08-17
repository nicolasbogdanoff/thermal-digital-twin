from __future__ import annotations

import numpy as np

from thermal_digital_twin import (
    ThermalParameters,
    fit_parameters,
    generate_observations,
)


time_s = np.linspace(0.0, 900.0, 181)
ambient_c = 25.0 + 1.5 * np.sin(time_s / 180.0)
power_w = np.where(time_s < 450.0, 180.0, 60.0)
true_parameters = ThermalParameters(ua=12.0, heat_capacity=2500.0)

observed_temperature_c = generate_observations(
    time_s,
    ambient_c,
    power_w,
    initial_temperature_c=25.0,
    params=true_parameters,
    noise_std_c=0.03,
    seed=7,
)

fitted_parameters, result = fit_parameters(
    time_s,
    ambient_c,
    power_w,
    observed_temperature_c,
    initial_temperature_c=25.0,
    initial_guess=(20.0, 4000.0),
)

print(f"Optimization successful: {result.success}")
print(f"True UA: {true_parameters.ua:.3f} | Fitted UA: {fitted_parameters.ua:.3f}")
print(
    "True heat capacity: "
    f"{true_parameters.heat_capacity:.3f} | "
    f"Fitted heat capacity: {fitted_parameters.heat_capacity:.3f}"
)
