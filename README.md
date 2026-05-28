# ✈️ Flight Price Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A machine learning application that predicts flight ticket prices for both **US Domestic** and **Indian Domestic** markets. By combining advanced predictive modeling with historical dataset anchoring, this tool provides highly accurate fare estimates, realistic price ranges, and actionable money-saving insights for travelers.

---

## 🌟 Key Features

- **Dual Market Support:** 
  - 🇺🇸 **US Domestic:** Supports 50+ major airports across the United States.
  - 🇮🇳 **Indian Domestic:** Supports 6 major metro cities (Delhi, Mumbai, Bangalore, Chennai, Kolkata, Hyderabad) and 75+ regional airports, with full support for Economy and Business class fares.
- **Hybrid AI Architecture:** 
  - Uses trained **XGBoost** pipelines for core price prediction based on distance, time to departure, airline tier, and seasonality.
  - **Dataset Anchoring:** Blends raw ML predictions with historical dataset averages to prevent wild model deviations and ensure highly realistic fare estimates.
- **Smart Money-Saving Tips:** Analyzes the route and booking window to suggest cheaper departure times, alternate budget airlines, or better booking windows.
- **Dataset Transparency:** Prints real dataset statistics (Average, Median, Min, Max fares) alongside the ML prediction so users can see exactly how the prediction compares to historical data.

---

## 📂 Project Structure

```text
FLIGHT PRICE PREDICTOR/
│
├── data/                       # Datasets
│   ├── raw/                    # Unprocessed data (e.g., Indian dataset)
│   └── processed/              # Cleaned data (e.g., US dataset)
│
├── models/                     # Serialized ML Models (.pkl, .joblib)
│   ├── flight_pipeline_v2.pkl
│   ├── indian_flight_best_pipeline.joblib
│   └── ... (encoders and feature columns)
│
├── notebooks/                  # Jupyter Notebooks for R&D
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_visualization.ipynb
│   ├── 05_indian_flight_analysis.ipynb
│   └── 06_indian_feature_eng_training.ipynb
│
├── src/                        # Source Code
│   ├── prediction.py           # Main CLI application
│   ├── preprocessing.py        # Data cleaning utilities
│   ├── training.py             # Model training scripts
│   └── utils.py                # Helper functions (geo-distance, etc.)
│
├── outputs/                    # Generated files (plots, logs)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Pip package manager

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/flight-price-predictor.git
cd flight-price-predictor
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Data Requirements
Due to file size constraints, the processed US dataset (`clean_dataset1.csv`) and Indian dataset (`clean_dataset.csv`) must be placed in the appropriate `data/` subdirectories prior to running the application.

---

## 🚀 Usage Guide

The application is run via a command-line interface (CLI). To start the predictor, execute:

```bash
python src/prediction.py
```

### Interactive Prompts
You will be asked to provide:
1. **Origin & Destination:** 3-letter IATA codes (e.g., `JFK`, `BOM`).
2. **Travel Date:** Format `YYYY-MM-DD`.
3. **Departure Time:** Format `HH:MM` (24-hour).
4. **Airline:** e.g., `Delta`, `IndiGo`, `United`.
5. **Days until Departure:** e.g., `14`.

*(For Indian routes, you will also be prompted for Flight Class and Number of Stops).*

### Example Output

```text
============================================================
📋 Flight Details:
   Route:       JFK → LAX (2469 miles)
   Date:        2024-12-15 (Sun)
   Departure:   08:30
   Airline:     Delta Air Lines Inc. (premium tier)
   Booking:     14 days before departure

────────────────────────────────────────────────────────────
💰 ML Predicted Fare:   $345.50
📊 Fare Range:          $295.00 — $410.00
────────────────────────────────────────────────────────────

📈 Prediction Breakdown:
   ML model base fare:     $320.00  (from trained pipeline)
   Booking adjustment:     14d ahead (×1.08)
   Final predicted fare:   $345.50

────────────────────────────────────────────────────────────
📊 DATASET PRICES (actual fares from your data):
────────────────────────────────────────────────────────────
   📍 Route JFK → LAX  (all airlines, 4,520 flights):
      Average:  $315.20
      Min:      $150.00
      Max:      $890.00
...
```

---

## 🧠 Model Architecture & Methodology

### Data Preprocessing
- **Geospatial Features:** Haversine distance calculated automatically based on IATA airport coordinate mappings.
- **Temporal Features:** Extraction of day of week, weekend flags, holiday seasonality, and departure time bucketing (Morning, Afternoon, Evening, etc.).
- **Target Encoding:** Complex categorical features (like specific routes) are target-encoded with Bayesian smoothing to prevent overfitting on low-frequency routes.

### Machine Learning Models
- **Algorithm:** XGBoost Regressor combined with Scikit-Learn Pipelines.
- **Evaluation:** Evaluated using Mean Absolute Error (MAE) and R² Score on a hold-out test set.
- **Post-Processing Calibration:** The raw ML output is blended with historical dataset medians. Fares are clamped within realistic boundaries derived directly from historical minimums and maximums for that specific airline and route.

---

## 🔮 Future Roadmap

- [ ] **Web Interface:** Build a frontend using Streamlit or FastAPI/React.
- [ ] **Live Pricing API Integration:** Integrate with Skyscanner or Amadeus APIs to compare ML predictions against live ticket prices.
- [ ] **Dynamic Pricing Alerts:** Allow users to set price alerts when the ML model predicts a massive impending price jump.
- [ ] **International Routes:** Expand the model to handle US-India cross-border pricing and European sectors.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
