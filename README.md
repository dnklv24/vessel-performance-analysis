# Vessel Performance Analysis & Optimization

## Overview
This repository contains the data engineering pipeline for a machine learning project aimed at predicting vessel shaft power, optimizing fuel efficiency, and reducing $CO_2$ emissions using real-world operational marine data (76,000 DWT bulk carrier). 

## My Role: Data Engineering
Within this collaborative academic project, my primary responsibility was designing and implementing a robust data processing architecture. I focused on transforming raw, noisy, high-frequency sensor telemetry into a clean, high-quality dataset ready for gradient boosting models (XGBoost/LightGBM).

## Tech Stack
* **Data Engineering (My Focus):** Python, `pandas`, `NumPy`
* **Machine Learning & UI (Project Context):** XGBoost, LightGBM, Streamlit

## Key Contributions & Pipeline Features
* **Advanced Telemetry Preprocessing:** Engineered solutions for multirate sampling asynchrony, handled missing values via asymmetric imputation (differentiating between inertial and dynamic signals), and corrected critical sensor anomalies (e.g., torque miscalibration and zero-value draft failures).
* **ISO 19030 Steady-State Filtering:** Implemented strict physical filtering criteria (e.g., speed > 6.0 knots, stable acceleration, shaft power thresholds) to isolate reliable steady-state transit data from erratic maneuvering periods.
* **Feature Engineering:** Designed and engineered physically relevant feature sets, including rolling averages for hydrodynamics (speed, wave height, draft), effective headwind vectors, and non-linear biological hull fouling proxies.

## Documentation
The complete project report detailing the physical foundation, machine learning model evaluation, and business impact analysis (including fuel expenditure and maintenance evaluations) is included in this repository as a PDF file.
