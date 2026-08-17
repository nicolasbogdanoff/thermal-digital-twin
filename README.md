# Thermal Digital Twin

A compact, reproducible Python demonstrator for a first-order thermal digital twin.

[![Tests](https://github.com/nicolasbogdanoff/thermal-digital-twin/actions/workflows/tests.yml/badge.svg)](https://github.com/nicolasbogdanoff/thermal-digital-twin/actions/workflows/tests.yml)

The repository connects a simple physical model with synthetic sensor data and parameter estimation. It is intended for engineering education, model-based reasoning, and transparent experimentation—not as a validated production model for a specific asset.

## Model

The model represents a lumped thermal system with temperature T, ambient temperature Ta, input power P, thermal conductance UA, and effective heat capacity C:

dT/dt = (P - UA(T - Ta)) / C

The implementation advances the state over a supplied time grid using the exact solution of the first-order model over each interval, with linearly interpolated input conditions.

## Features

- Deterministic simulation of a first-order thermal system.
- Time-varying ambient temperature and input-power signals.
- Synthetic observations with reproducible Gaussian sensor noise.
- Least-squares estimation of UA and C from observed temperature data.
- Input validation for time grids, physical parameters, and signal shapes.
- Small unit-test suite covering simulation and parameter recovery.

## Quick start

~~~bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

pip install -e ".[dev]"
pytest
~~~

## Example

~~~bash
python examples/basic_simulation.py
~~~

The example also illustrates the public API:

~~~python
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
~~~

## Interpreting the parameters

- UA controls how strongly the system exchanges heat with its surroundings.
- C controls how quickly the temperature responds to changes in power or ambient conditions.
- Different combinations of UA and C can produce similar short-term responses. Identifiability depends on excitation, sampling duration, noise, and the observed input signals.

## Repository layout

| Path | Purpose |
| --- | --- |
| thermal_digital_twin/model.py | Simulation, synthetic observations, and parameter fitting |
| thermal_digital_twin/__init__.py | Public package interface |
| tests/test_model.py | Unit tests |
| examples/basic_simulation.py | Runnable example |
| .github/workflows/tests.yml | Continuous integration |
| CITATION.cff | Citation metadata |
| LICENSE | MIT license |
| pyproject.toml | Package metadata and dependencies |

## Scope and limitations

This is a deliberately small demonstrator. It assumes a lumped thermal state, does not model spatial gradients or phase changes, and does not claim calibration against a real system. Any engineering deployment should establish the model structure, parameters, sensor quality, boundary conditions, and validation evidence for the target asset.

## Relation to the profile

This repository is an initial, evidence-based project in digital twins and thermal engineering. It is designed to grow toward richer heat-transfer models, uncertainty quantification, and integration with measured data while keeping each extension inspectable.

## Author

Nicolás Mauricio Bogdanoff  
ORCID: https://orcid.org/0009-0004-6275-3013

## Citation

If this software contributes to a technical report, class activity, research note, or publication, cite it using [CITATION.cff](CITATION.cff).

## License

This project is released under the [MIT License](LICENSE).
