import numpy as np
import pytest

from thermal_digital_twin import (
    ThermalParameters,
    fit_parameters,
    generate_observations,
    simulate_temperature,
)


def test_constant_inputs_approach_the_expected_equilibrium() -> None:
    time = np.linspace(0.0, 1200.0, 241)
    params = ThermalParameters(ua=12.0, heat_capacity=2500.0)

    temperature = simulate_temperature(
        time,
        ambient_c=25.0,
        power_w=180.0,
        initial_temperature_c=25.0,
        params=params,
    )

    expected_equilibrium = 25.0 + 180.0 / 12.0
    assert abs(temperature[-1] - expected_equilibrium) < 0.2


def test_parameter_estimation_recovers_a_noisy_synthetic_system() -> None:
    time = np.linspace(0.0, 900.0, 181)
    ambient = 25.0 + 1.5 * np.sin(time / 180.0)
    power = np.where(time < 450.0, 180.0, 60.0)
    true_parameters = ThermalParameters(ua=12.0, heat_capacity=2500.0)

    observed = generate_observations(
        time,
        ambient,
        power,
        initial_temperature_c=25.0,
        params=true_parameters,
        noise_std_c=0.03,
        seed=7,
    )

    fitted, result = fit_parameters(
        time,
        ambient,
        power,
        observed,
        initial_temperature_c=25.0,
        initial_guess=(20.0, 4000.0),
    )

    assert result.success
    assert fitted.ua == pytest.approx(true_parameters.ua, rel=0.2)
    assert fitted.heat_capacity == pytest.approx(true_parameters.heat_capacity, rel=0.2)


def test_invalid_time_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        simulate_temperature(
            [0.0, 1.0, 1.0],
            ambient_c=25.0,
            power_w=100.0,
            initial_temperature_c=25.0,
            params=ThermalParameters(ua=10.0, heat_capacity=1000.0),
        )
