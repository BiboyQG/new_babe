# Time Series Forecasting Models

This directory contains implementation of various time series forecasting models for predicting `gap_open` values.

## Models

### ARIMA

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

## Usage

To train and evaluate all ARIMA models:

```python
python models/arima_model.py
```
