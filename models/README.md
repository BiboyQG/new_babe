# Time Series Forecasting Models

This directory contains implementation of various time series forecasting models for predicting `gap_open` values.

## Models

### 1.ARIMA

The ARIMA (AutoRegressive Integrated Moving Average) model is implemented in `arima_model.py`. This model uses exogenous variables:

- `inflation_index`
- `interest_OPEN` (CN datasets) / `interest_Rate (%)` (US datasets)
- `oil_CLOSE`

#### Results

| Region | Period          | Mean Absolute Error | Root Mean Squared Error | Average Difference |
| ------ | --------------- | ------------------- | ----------------------- | ------------------ |
| CN     | 2023.01-2023.12 | 3.885061            | 5.468841                | 3.885061           |
| CN     | 2023.06-2024.06 | 1.491706            | 2.067563                | 1.491706           |
| US     | 2023.01-2023.12 | 1.772954            | 2.304969                | 1.772954           |
| US     | 2023.06-2024.06 | 2.957172            | 3.717040                | 2.957172           |

#### Model Details

- ARIMA order: (2, 1, 2) (p, d, q parameters)
- Trained individually for each region and time period
- Uses SARIMAX implementation from the `statsmodels` package

For each region and period, the following outputs are generated:

- `models/arima/{region}/{period}/arima_model_{region}_{period}.pkl`: Saved model
- `models/arima/{region}/{period}/predictions_{region}_{period}.csv`: Predictions on validation data
- `models/arima/{region}/{period}/plot_{region}_{period}.png`: Plot of actual vs predicted values
- `models/arima/{region}/{period}/results_summary.txt`: Summary of model performance

### 2. XGBoost

The XGBoost (eXtreme Gradient Boosting) model is implemented in `xgboost_model.py`. This model uses the same exogenous variables:

- `inflation_index`
- `interest_OPEN` (CN datasets) / `interest_Rate (%)` (US datasets)
- `oil_CLOSE`

#### Results

| Region | Period          | Mean Absolute Error | Root Mean Squared Error | Average Difference |
| ------ | --------------- | ------------------- | ----------------------- | ------------------ |
| CN     | 2023.01-2023.12 | 3.277267            | 4.172293                | 3.277267           |
| CN     | 2023.06-2024.06 | 2.901246            | 3.434522                | 2.901246           |
| US     | 2023.01-2023.12 | 4.529064            | 4.971531                | 4.529064           |
| US     | 2023.06-2024.06 | 6.469616            | 7.491910                | 6.469616           |

#### Model Details

- XGBoost parameters:
  - objective: reg:squarederror
  - learning_rate: 0.05
  - max_depth: 6
  - min_child_weight: 2
  - subsample: 0.8
  - colsample_bytree: 0.8
  - n_estimators: 200
  - random_state: 42
- Trained individually for each region and time period
- Uses XGBRegressor from the `xgboost` package

For each region and period, the following outputs are generated:

- `models/xgboost/{region}/{period}/xgboost_model_{region}_{period}.pkl`: Saved model
- `models/xgboost/{region}/{period}/predictions_{region}_{period}.csv`: Predictions on validation data
- `models/xgboost/{region}/{period}/plot_{region}_{period}.png`: Plot of actual vs predicted values
- `models/xgboost/{region}/{period}/results_summary.txt`: Summary of model performance
- `models/xgboost/{region}/{period}/feature_importance_{region}_{period}.csv`: Feature importance analysis

### 3. LSTM

The LSTM (Long Short-Term Memory) model is implemented in `lstm_model.py`. This deep learning sequence model uses the same exogenous variables as the previous models:

- `inflation_index`
- `interest_OPEN` (CN datasets) / `interest_Rate (%)` (US datasets)
- `oil_CLOSE`

#### Results

| Region | Period          | Mean Absolute Error | Root Mean Squared Error | Average Difference |
| ------ | --------------- | ------------------- | ----------------------- | ------------------ |
| CN     | 2023.01-2023.12 | 12.200084           | 15.265879               | 12.200084          |
| CN     | 2023.06-2024.06 | 8.574166            | 9.322096                | 8.574166           |
| US     | 2023.01-2023.12 | 3.413341            | 4.250748                | 3.413341           |
| US     | 2023.06-2024.06 | 7.033762            | 8.575963                | 7.033762           |

#### Model Details

- Architecture:

  - LSTM layers: 2
  - Hidden size: 64
  - Sequence length: 60
  - Dropout: 0.2
  - Output: 1 (gap_open prediction)

- Training parameters:

  - Batch size: 64
  - Learning rate: 0.001
  - Optimizer: Adam
  - Loss function: MSE (Mean Squared Error)
  - Epochs: 100
  - GPU acceleration: MPS (Metal Performance Shaders) on macOS

- Data preprocessing:
  - Min-Max scaling for all features and target
  - Sequence preparation with sliding window approach
  - PyTorch DataLoader with batching and shuffling

For each region and period, the following outputs are generated:

- `models/lstm/{region}/{period}/lstm_model_{region}_{period}.pt`: Saved PyTorch model with scalers
- `models/lstm/{region}/{period}/predictions_{region}_{period}.csv`: Predictions on validation data
- `models/lstm/{region}/{period}/plot_{region}_{period}.png`: Plot of actual vs predicted values
- `models/lstm/{region}/{period}/training_loss_{region}_{period}.png`: Training loss curve
- `models/lstm/{region}/{period}/results_summary.txt`: Summary of model performance and hyperparameters

## Usage

To train and evaluate all ARIMA models:

```python
python models/arima_model.py
```

To train and evaluate all XGBoost models:

```python
python models/xgboost_model.py
```

To train and evaluate all LSTM models:

```python
python models/lstm_model.py
```
