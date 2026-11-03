# Bike Demand Category Predictor

An urban-mobility machine-learning mini-project that classifies bike-sharing demand as low, medium, or high from weather, hour, and weekend indicators.

## Overview

Bike-sharing operators can use demand categories to plan rebalancing and staffing. This project implements a transparent nearest-centroid classifier that summarizes each category and assigns new observations to the closest demand profile.

## Problem statement

Given temperature, humidity, hour, and weekend status, predict an interpretable bike-demand category.

## Features

- Nearest-centroid classification from scratch.
- Multivariate Euclidean distance.
- Reproducible CSV data and CLI.
- Training accuracy and unit tests.
- Standard-library-only implementation.

## Technology stack

- Python 3.10+
- Nearest-centroid classification
- CSV, `argparse`, and `pathlib`
- `unittest` for automated testing

## Installation

```bash
git clone https://github.com/mohan843/bike-demand-category-predictor.git
cd bike-demand-category-predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

No third-party runtime packages are required.

## Execution

```bash
python predictor.py --temperature 22 --humidity 50 --hour 9 --is-weekend 0
```

## Testing

```bash
python -m unittest discover -s tests -v
```

## Sample output

```text
Training accuracy: 60.0%
Predicted bike demand: high
```

## Limitations

The sample data is synthetic and small. Real mobility systems should include station capacity, events, holidays, precipitation, wind, route structure, and time-aware validation.

## Future improvements

- Predict station-level demand rather than a global category.
- Compare with random forests and gradient boosting.
- Add confidence estimates and calibrated probabilities.
- Include live weather and event signals.
- Build a rebalancing dashboard for operators.
