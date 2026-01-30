# JobAI Salary Predictor

A machine learning pipeline designed to predict AI-related job salaries based on industry data, experience levels, and skill demands. This project uses a **Random Forest Regressor** to achieve high-precision salary estimates.

## Performance Metrics
* **Mean Absolute Error (MAE):** $3,546.74
* **R2 Score (Accuracy):** 0.9867 (98.6%)

---

## Project Structure
```text
JobAI/
├── data/               # Raw CSV datasets
├── models/             # Saved .pkl model files
├── src/                # Source code
│   ├── data_loader.py   # Data cleaning & ID normalization
│   ├── preprocessing.py  # Encoding & Scaling logic
│   ├── train.py         # Model training script
│   ├── evaluate.py      # Feature importance analysis
│   └── predict.py       # Single-point inference
└── main.py             # Project entry point

![AI Salary Feature Importance](salary_features.png)
