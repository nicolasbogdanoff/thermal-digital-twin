from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class ThermalParameters:
    """Physical parameters for the lumped first-order thermal model."""

    ua: float
    heat_capacity: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.ua) or self.ua <= 0:
            raise ValueError("ua must be finite and greater than zero.")
        if not np.isfinite(self.heat_capacity) or self.heat_capacity <= 0:
            raise ValueError("heat_capacity must be finite and greater than zero.")


def _time_grid(time_s: Any) -> np.ndarray:
    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time_s must be a one-dimensional array with at least two points.")
    if not np.isfinite(time).all() or not np.all(np.diff(time) > 0):
        raise ValueError("time_s must be finite and strictly increasing.")
    return time


def _signal(value: Any, size: int, name: str) -> np.ndarray:
    signal = np.broadcast_to(np.asarray(value, dtype=float), (size,))
    if not np.isfinite(signal).all():
        raise ValueError(f"{name} must contain only finite values.")
    return signal


def simulate_temperature(
    time_s: Any,
    ambient_c: Any,
    power_w: Any,
    initial_temperature_c: float,
    params: ThermalParameters,
) -> np.ndarray:
    """Simulate temperature for scalar or sampled ambient and power inputs."""

    time = _time_grid(time_s)
    ambient = _signal(ambient_c, time.size, "ambient_c")
    power = _signal(power_w, time.size, "power_w")
    initial = float(initial_temperature_c)
    if not np.isfinite(initial):
        raise ValueError("initial_temperature_c must be finite.")

    temperature = np.empty_like(time)
    temperature[0] = initial

    for index, delta_t in enumerate(np.diff(time), start=1):
        ambient_mid = 0.5 * (ambient[index - 1] + ambient[index])
        power_mid = 0.5 * (power[index - 1] + power[index])
        equilibrium = ambient_mid + power_mid / params.ua
        decay = np.exp(-params.ua * delta_t / params.heat_capacity)
        temperature[index] = equilibrium + (
            temperature[index - 1] - equilibrium
        ) * decay

    return temperature


def generate_observations(
    time_s: Any,
    ambient_c: Any,
    power_w: Any,
    initial_temperature_c: float,
    params: ThermalParameters,
    noise_std_c: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate deterministic synthetic temperature observations."""

    noise_std = float(noise_std_c)
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std_c must be finite and non-negative.")

    clean = simulate_temperature(
        time_s,
        ambient_c,
        power_w,
        initial_temperature_c,
        params,
    )
    rng = np.random.default_rng(seed)
    return clean + rng.normal(0.0, noise_std, size=clean.size)


def fit_parameters(
    time_s: Any,
    ambient_c: Any,
    power_w: Any,
    observed_temperature_c: Any,
    initial_temperature_c: float,
    initial_guess: tuple[float, float] = (10.0, 2500.0),
) -> tuple[ThermalParameters, Any]:
    """Estimate UA and heat capacity with bounded-positive least squares."""

    observed = np.asarray(observed_temperature_c, dtype=float)
    time = _time_grid(time_s)
    if observed.shape != time.shape or not np.isfinite(observed).all():
        raise ValueError("observed_temperature_c must match time_s and be finite.")

    guess_ua, guess_capacity = map(float, initial_guess)
    if guess_ua <= 0 or guess_capacity <= 0:
        raise ValueError("initial_guess values must be greater than zero.")

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        fitted = ThermalParameters(
            ua=float(np.exp(log_parameters[0])),
            heat_capacity=float(np.exp(log_parameters[1])),
        )
        predicted = simulate_temperature(
            time,
            ambient_c,
            power_w,
            initial_temperature_c,
            fitted,
        )
        return predicted - observed

    result = least_squares(
        residual,
        x0=np.log([guess_ua, guess_capacity]),
    )
    fitted_parameters = ThermalParameters(
        ua=float(np.exp(result.x[0])),
        heat_capacity=float(np.exp(result.x[1])),
    )
    return fitted_parameters, result
