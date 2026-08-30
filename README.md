# Used Car Price Prediction

An end-to-end machine learning project that predicts the resale price of used cars based on features like manufacturer, model year, odometer reading, condition, fuel type, and more — trained on a real-world dataset of over 400,000 Craigslist vehicle listings.

## Overview

Used car pricing is notoriously inconsistent — identical models can be listed at very different prices depending on condition, mileage, and seller behavior. This project builds a regression pipeline to predict a fair market price from a vehicle's listed features, and compares nine different ML algorithms to find the best-performing approach.

## Dataset

- **Source:** [Craigslist Cars & Trucks Data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) (Kaggle, by Austin Reese)
- **Size:** 426,880 rows × 26 columns (raw)
- **Target variable:** `price`

## Project Workflow

1. **Data Understanding** — inspected structure, data types, and missing values across all 26 original columns
2. **Data Cleaning & Feature Engineering**
   - Dropped identifier/leakage-prone columns (`id`, `url`, `VIN`, `image_url`, `description`, `lat`, `long`, `county`)
   - Ordinal encoding for `condition` and `size` (preserves the natural ranking, e.g. fair < good < excellent)
   - One-hot encoding for nominal categorical features (`manufacturer`, `type`, `fuel`, `transmission`, `drive`, `title_status`, `paint_color`)
   - Engineered `car_age` from `posting_date` and `year`
   - Handled missing values with a mix of median imputation, `"unknown"` category fill, and row-dropping depending on the column
   - Removed price outliers using the IQR method
3. **Model Training & Evaluation**
   - Train/test split (75/25)
   - Feature scaling with `StandardScaler`
   - Trained and compared 9 regression models
   - Hyperparameter tuning on the top performer using `RandomizedSearchCV`

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 7,650.37 | 10,402.36 | 0.367 |
| Ridge | 7,647.86 | 10,402.44 | 0.367 |
| Lasso | 7,648.12 | 10,402.37 | 0.367 |
| ElasticNet | 7,913.24 | 10,519.34 | 0.353 |
| AdaBoost | 7,601.01 | 10,081.50 | 0.405 |
| Gradient Boosting | 6,134.42 | 8,994.67 | 0.527 |
| XGBoost | 5,164.80 | 7,913.84 | 0.634 |
| Decision Tree | 3,162.52 | 7,402.29 | 0.679 |
| **Random Forest** | **2,816.86** | **5,780.99** | **0.804** |

**Random Forest was the best-performing model**, explaining ~80% of the variance in used car prices with a mean absolute error of roughly $2,800 — tree-based ensemble methods significantly outperformed linear models, suggesting non-linear relationships between features like odometer reading, age, and price.

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn

## Repository Structure

```
├── Notebooks/
│   ├── Data_Understanding.ipynb                   # Initial data exploration
│   ├── Data_Cleaning_and_Feature_Engineering.ipynb # Cleaning, encoding, EDA
│   └── Model_Training.ipynb                       # Model comparison & tuning
├── Dataset/                                       # Raw and cleaned data
├── Images/                                        # Saved plots/visualizations
└── README.md
```

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/keshvaniabbasali-cmd/Car-Price-Prediction.git
   cd Car-Price-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) and place `vehicles.csv` in the `Dataset/` folder
4. Run the notebooks in order: `Data_Understanding` → `Data_Cleaning_and_Feature_Engineering` → `Model_Training`

## Key Learnings

- Tree-based ensemble methods handle the non-linear, high-cardinality nature of vehicle pricing data far better than linear models
- Careful handling of missing data and encoding strategy (ordinal vs. one-hot) has a meaningful impact on model performance
- Feature engineering (e.g., deriving `car_age`) can surface relationships that raw columns don't capture directly

## Future Improvements

- Fix train/test leakage in outlier removal (currently applied before the split)
- Add feature importance analysis to identify the strongest price drivers
- Save the final trained model with `joblib` for deployment
- Wrap preprocessing + model into a single `sklearn` Pipeline for reproducibility
- Deploy as a simple web app (Flask/Streamlit) for interactive predictions

## Author

**Abbasali Keshvani**
[LinkedIn](https://linkedin.com/in/abbasali-keshvani-425606328) · [GitHub](https://github.com/keshvaniabbasali-cmd)
