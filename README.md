# Hotel No-Show Prediction

## Project Description
This project implements an end to end machine learning pipeline to predict the No-Show of customers for a hotel chain.  The pipeline includes data cleaning, preprocessing, model training and evaluation. Multiple classification models are trained and compared to find the best performing one based on various metrics. 
## Prerequisites and Installation Instructions

### Requirements
- Anaconda or Miniconda
- Python 3.11
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/taenyfan/predict-hotel-no-show.git
   cd predict-hotel-no-show
   ```

2. Create and activate a conda environment:
   ```
   conda create -n hdbenv python=3.11
   conda activate hdbenv
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Pipeline Execution
To run the complete pipeline, execute:
```
python main.py
```

This will:
1. Load and clean the data
2. Preprocess features
3. Split data into training, validation and test sets
4. Train baseline models (including a dummy classifier)
5. Perform hyperparameter tuning
6. Evaluate models and select the best one
7. Evaluate the best model on the test set

To run the unit tests for the data preparation functions:
```
cd src
pytest test_data_preparation.py -v
```

## Logical Flow of the Pipeline

### 1. Configuration (`config.yaml`)
The pipeline starts with loading configuration parameters from `src/config.yaml`, which includes:
- Data file path
- Target column name
- Feature categorization (numerical, nominal, cyclical)
- Features to drop
- Train/validation/test split ratios
- Hyperparameter grids for model tuning
- USD-to-SGD exchange rate 


### 2. Data Preparation (`DataPreparation` class)
The `DataPreparation` class handles:
- Dropping the fully-null row
- Standardizing month name casing
- Fixing negative checkout days via absolute value
- Parsing `num_adults` (words mixed with digits) to integers
- Converting `price` from dual-currency strings to numeric SGD
- Imputing missing `price` from the (branch, room) median
- Imputing missing `room` from branch + price band
- Engineering `lead_time` (months between booking and arrival)
- Dropping low/no-signal columns and the row identifier
- Creating a preprocessor for feature transformation

### 3. Model Training (`ModelTraining` class)
The `ModelTraining` class manages:
- Splitting data into stratified training, validation and test sets
- Training baseline models with default hyperparameters (Dummy, Logistic Regression, KNN, Random Forest)
- Hyperparameter tuning via GridSearchCV for Logistic Regression, KNN and Random Forest
- Evaluating models using multiple metrics (accuracy, precision, recall, F1, ROC-AUC)

### 4. Main Execution (`main.py`)
The main script orchestrates the entire pipeline by:
- Loading configuration and data
- Initializing data preparation and model training
- Running the training and evaluation process
- Identifying the best model based on validation ROC-AUC
- Evaluating the best model on the held-out test set

## Key Findings and Explanations

### Exploratory Data Analysis

Before model training, an exploratory data analysis was conducted to understand the data and surface issues that would shape the preprocessing pipeline.

**EDA Findings and Explanations**

1. **Fully-null row in raw data**
   The raw dataset contained 119,391 rows, one of which was **entirely null across every column except `booking_id`** — a junk record with no usable information, including no target label. This row was dropped, leaving 119,390 valid bookings. 

2. **Mild class imbalance in the target**
   The target `no_show` is split roughly **63% shows (0) / 37% no-shows (1)**. This is only a mild imbalance, so no resampling was applied — the 44,224 minority examples are ample. However, the imbalance does mean **accuracy is misleading** (a majority-class predictor scores ~63% while catching zero no-shows), which is why ROC-AUC  is used as the primary metric rather than accuracy.

3. **Dual-currency `price` strings**
   The `price` column was stored as strings mixing two currencies, e.g. `"SGD$ 450"` and `"USD$ 320"`, in roughly equal proportion. A parser extracts the currency and numeric amount, then converts USD entries to SGD using an exchange rate of 1.37 (recovered empirically from the data). This standardises all prices to a single currency before modelling.

4. **Missing values in `price` and `room`**
   `price` had 24,882 missing values and `room` had 21,613. These were imputed using the structural relationship between branch, room, and price discovered in EDA: price is determined largely by branch and room type. Missing `price` is filled with the median price of its (branch, room) group. Missing `room` is imputed by checking which room's price band (1st–99th percentile, per branch) the booking's price falls into; ambiguous or out-of-range cases are assigned `"Unknown"` rather than guessed.

6. **Duplicate rows are coincidental, not errors**
   The dataset contained 1,514 feature-identical rows. Unlike a row-identifier-based duplicate check, these are bookings that genuinely share the same feature values by coincidence (the dataset has many categorical features with few levels). They were **kept**, as they represent legitimate distinct bookings rather than data-entry errors.

7. **Inconsistent month casing**
   `arrival_month` contained casing-corrupted entries (e.g. `"january"`, `"JANUARY"`). These were standardised with `.str.title()` so each month is treated as a single category rather than several.

8. **Negative `checkout_day` values**
   `checkout_day` contained physically-impossible negative values, which were corrected by taking the absolute value rather than dropping the rows.

9. **`num_adults` mixed words and digits**
   `num_adults` contained entries like `"one"` and `"two"` alongside numeric values. These were mapped to integers (`one` → 1, `two` → 2) so the column could be treated as numeric. (This column was ultimately dropped — see finding 12.)

10. **`lead_time` is a strong engineered predictor**
    A `lead_time` feature (months between booking and arrival) was engineered from `booking_month` and `arrival_month`, with year-wrap handled (e.g. booked December, arriving February → 2 months). The no-show rate rises monotonically with lead time — from ~25% for same-month bookings to ~43% for bookings made ~10 months ahead — making it one of the most informative features. `booking_month` is dropped after `lead_time` is computed, as its signal is captured by the new feature.

11. **`country` is the single most predictive feature**
    Mutual-information analysis and Random Forest feature importance both rank `country` highest, followed by `price` and `lead_time`. Notably, the Random Forest importances show `price` contributes **independent** signal (ranking 2nd) rather than being merely a proxy for branch and room — within-category price variation relates to no-show behaviour beyond what the branch/room categories encode.

12. **Low/no-signal features dropped**
    Based on mutual-information and bivariate analysis, several features showed negligible association with the target and were dropped: `platform`, `num_children`, `num_adults`, `checkout_day`, and `arrival_day`. `booking_id` (a row identifier) and `booking_month` (consumed by `lead_time`) were also dropped.

### Data Cleaning
- The fully-null junk row is dropped because it contains no usable information, including no target label.
- `price` strings are parsed and converted to a single currency (SGD) because the raw column mixes USD and SGD, which the model cannot interpret as-is.
- Missing `price` and `room` are imputed from the branch/room/price structure because price is structurally determined by branch and room, making group-based imputation reliable.
- `arrival_month` casing is standardised, `checkout_day` is made positive, and `num_adults` words are converted to digits because these are data-entry inconsistencies that would otherwise split or corrupt the affected columns.
- `lead_time` is engineered before `booking_month` is dropped because it is computed from the booking and arrival months and carries stronger signal than either alone.

### Feature Processing
- **Numerical Features**: `lead_time`, `price` — Standardized using `StandardScaler` for the distance- and coefficient-based models (KNN, Logistic Regression), which are sensitive to feature scale. Scaling is skipped for the tree-based Random Forest, which is scale-invariant.
- **Nominal Features**: `branch`, `country`, `first_time`, `room` — Encoded using `OneHotEncoder` with `handle_unknown='ignore'` because they have no intrinsic order, and the `ignore` setting prevents unseen categories in validation/test from crashing inference.
- **Cyclical Features**: `arrival_month`, `checkout_month` — Encoded as sine/cosine pairs because months are cyclical (December is adjacent to January); integer encoding would wrongly place them 11 apart, whereas sin/cos encoding preserves their circular adjacency.

`room` was encoded as nominal (one-hot) rather than ordinal despite having a natural size order, because (a) the imputed `"Unknown"` category has no position on a size scale, and (b) the no-show rate is non-monotonic with room tier, so an ordinal encoding would mislead the linear model.

## Model Choices and Evaluation

### Models Implemented and Justifications
1. **Dummy Classifier** (`most_frequent`): A no-skill baseline that always predicts the majority class (show). It establishes the floor that every real model must beat — it scores ~63% accuracy but ROC-AUC 0.50 and F1 0, illustrating why accuracy alone is misleading on this imbalanced data.

2. **Logistic Regression**: A linear, interpretable baseline classifier. It is fast, provides a probabilistic output, and serves as a strong reference point for the more complex models.

3. **K-Nearest Neighbours (KNN)**: A non-parametric, distance-based classifier that can capture local structure in the feature space without assuming a functional form.

4. **Random Forest**: An ensemble of decision trees (bagging + per-split feature randomness). It captures non-linear relationships and feature interactions, is robust without heavy tuning, handles mixed feature types well, and provides feature importances. It is chosen over a single decision tree (lower variance) and over plain bagged trees (the feature randomness decorrelates the trees, making the averaging more effective).

### Hyperparameter Tuning
- Grid search (`GridSearchCV`) is performed for Logistic Regression, KNN and Random Forest because their default hyperparameters are rarely optimal, and an exhaustive search over a sensible grid finds the best configuration systematically.
- Parameters tuned: Logistic Regression — `C` (regularization strength) and `penalty` (L1/L2); KNN — `n_neighbors`, `weights`, and `metric`; Random Forest — `n_estimators`, `max_depth`, and `min_samples_leaf`.
- 5-fold cross-validation is used during tuning because evaluating each hyperparameter combination on 5 held-out folds (rather than a single split) gives a more reliable estimate that is not at the mercy of one lucky or unlucky partition.
- `scoring='roc_auc'` is used as the tuning objective because it is threshold-independent and robust to the class imbalance.

### Evaluation Metrics
- **Accuracy**: Proportion of correct predictions. Reported for context, but misleading on imbalanced data (a majority-class guess scores ~63%).
- **Precision**: Of the bookings predicted as no-shows, the proportion that actually were. Measures false-alarm cost.
- **Recall**: Of the actual no-shows, the proportion the model caught. Measures missed no-shows.
- **F1**: The harmonic mean of precision and recall — a single balanced measure of minority-class performance.
- **ROC-AUC**: The primary selection metric. Measures how well the model ranks no-shows above shows across all thresholds; threshold-independent and robust to imbalance.

### Results
On the validation set, all three tuned models beat the dummy baseline, with Random Forest performing best on the primary metric:

| Model               | Accuracy | Precision | Recall  | F1      | ROC-AUC |
|---------------------|----------|-----------|---------|---------|---------|
| Dummy (baseline)    | 0.6295   | 0.0000    | 0.0000  | 0.0000  | 0.5000  |
| Logistic Regression | 0.7266   | 0.6659    | 0.5259  | 0.5877  | 0.7507  |
| KNN                 | 0.7381   | 0.6988    | 0.5150  | 0.5930  | 0.7787  |
| Random Forest       | 0.7422   | 0.7046    | 0.5236  | 0.6008  | 0.7831  |

Random Forest was selected as the best model (highest validation ROC-AUC) and evaluated once on the held-out test set, achieving **ROC-AUC 0.7855** and **F1 0.6032** — closely matching its validation performance, confirming the model generalises and was not overfit during selection. All models are precision-favouring at the default 0.5 threshold (catching ~53% of no-shows while rarely false-flagging confirmed stays); recall could be raised by lowering the decision threshold if catching more no-shows were prioritised.

## Testing
Unit tests for the data preparation functions are in `src/test_data_preparation.py` and run with `pytest`. They cover the cleaning and feature-engineering logic — currency conversion, the (branch, room) imputations, `lead_time` (including year-wrap edge cases), and the cyclical month encoding — verifying both normal and edge-case behaviour.

## Limitations and Future Work
1. **Model expressiveness**: Performance plateaus around ROC-AUC 0.78. Gradient-boosting models (XGBoost, LightGBM) typically outperform Random Forest on tabular data and could be explored.
2. **Feature engineering**: Additional features such as interaction terms (e.g. country × lead_time), finer temporal features (day-of-week, holiday flags), or historical no-show rates per segment may add signal beyond what the current features capture.
3. **Threshold tuning**: The default 0.5 decision threshold favours precision. If the business goal were to catch more no-shows, the threshold could be tuned on the validation set (optimising F2, which weights recall more heavily) to trade precision for recall.
4. **Irreducible noise**: No-show behaviour depends on factors outside the booking data (personal circumstances, weather, competing offers), placing a natural ceiling on achievable performance; a ROC-AUC near 0.78 is consistent with this and is not a sign of an under-performing model.
