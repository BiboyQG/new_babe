import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_dataset(file_path):
    """Load a dataset from a CSV file"""
    return pd.read_csv(file_path)


def prepare_data_for_arima(
    df, target_col="gap_open", datetime_col="DateTime", exog_cols=None
):
    """
    Prepare data for ARIMA model with exogenous variables

    Parameters:
    - df: DataFrame containing the data
    - target_col: Target column to predict
    - datetime_col: Column containing datetime information
    - exog_cols: List of exogenous variables to include

    Returns:
    - endog: pandas Series with DateTime index (target variable)
    - exog: pandas DataFrame with DateTime index (exogenous variables)
    """
    # Make a copy of the dataframe
    df_copy = df.copy()

    # Convert datetime to index
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])
    df_copy = df_copy.set_index(datetime_col)

    # Extract the target column as the endogenous variable
    endog = df_copy[target_col]

    # Extract exogenous variables if provided
    exog = None
    if exog_cols:
        exog = df_copy[exog_cols]

    return endog, exog


def train_arima_model(endog, exog=None, order=(1, 1, 1), seasonal_order=None):
    """
    Train ARIMA model with or without exogenous variables

    Parameters:
    - endog: Target time series
    - exog: Exogenous variables (optional)
    - order: ARIMA order (p, d, q)
    - seasonal_order: Seasonal order (P, D, Q, s) if needed

    Returns:
    - model_result: Fitted ARIMA model
    """
    if seasonal_order:
        model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order)
    else:
        model = SARIMAX(endog, exog=exog, order=order)

    print(f"Training ARIMA model with order {order}...")
    model_result = model.fit(disp=False)
    print("Model training complete.")

    return model_result


def evaluate_model(model, test_endog, test_exog=None):
    """
    Evaluate ARIMA model on test data

    Parameters:
    - model: Fitted ARIMA model
    - test_endog: Test target variable
    - test_exog: Test exogenous variables (optional)

    Returns:
    - predictions: Predicted values
    - mae: Mean Absolute Error
    - rmse: Root Mean Squared Error
    """
    # Generate predictions
    predictions = model.get_forecast(
        steps=len(test_endog), exog=test_exog
    ).predicted_mean

    # Calculate error metrics
    mae = mean_absolute_error(test_endog, predictions)
    rmse = np.sqrt(mean_squared_error(test_endog, predictions))

    return predictions, mae, rmse


def save_model(model, file_path):
    """
    Save the trained model to disk

    Parameters:
    - model: Trained model to save
    - file_path: Path to save the model
    """
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")


def load_model(file_path):
    """
    Load a saved model from disk

    Parameters:
    - file_path: Path to the saved model

    Returns:
    - model: Loaded model
    """
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model


def plot_results(actual, predicted, title, save_path=None):
    """
    Plot actual vs predicted values

    Parameters:
    - actual: Actual values
    - predicted: Predicted values
    - title: Plot title
    - save_path: Path to save the plot (optional)
    """
    # Create a DataFrame with both actual and predicted values using the same index
    results_df = pd.DataFrame({"Actual": actual})

    # Assign the DatetimeIndex from actual to predicted values
    results_df["Predicted"] = predicted.values
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(results_df.index, results_df["Actual"], label="Actual")
    plt.plot(results_df.index, results_df["Predicted"], label="Predicted", color="red")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("gap_open")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    # Disable interactive display
    plt.close()


def run_arima_experiment(region, period, base_dir="prepared_data", models_dir="models"):
    """
    Run complete ARIMA experiment for a given region and period

    Parameters:
    - region: 'CN' or 'US'
    - period: '2023.01-2023.12' or '2023.06-2024.06'
    - base_dir: Directory containing the data
    - models_dir: Directory to save models and results

    Returns:
    - results: Dictionary containing evaluation metrics
    """
    # Define file paths
    filename_prefix = f"{region}_{period.replace('.', '').replace('-', '_')}"
    train_file = os.path.join(base_dir, f"{filename_prefix}_train.csv")
    val_file = os.path.join(base_dir, f"{filename_prefix}_val.csv")

    # Create output directory if it doesn't exist
    results_dir = os.path.join(
        models_dir, "arima", region, period.replace(".", "").replace("-", "_")
    )
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nRunning ARIMA experiment for {region} {period}...")

    # Load datasets
    train_df = load_dataset(train_file)
    val_df = load_dataset(val_file)

    # Determine exogenous variable column names (handle slight difference in column naming between CN and US datasets)
    if region == "CN":
        exog_cols = ["inflation_index", "interest_OPEN", "oil_CLOSE"]
    else:
        exog_cols = ["inflation_index", "interest_Rate (%)", "oil_CLOSE"]

    # Prepare data
    train_endog, train_exog = prepare_data_for_arima(
        train_df, target_col="gap_open", exog_cols=exog_cols
    )
    val_endog, val_exog = prepare_data_for_arima(
        val_df, target_col="gap_open", exog_cols=exog_cols
    )

    # Train model
    # We'll try a simple ARIMA model first
    arima_order = (2, 1, 2)  # p, d, q parameters (can be tuned)
    model = train_arima_model(train_endog, train_exog, order=arima_order)

    # Save model
    model_path = os.path.join(
        results_dir,
        f"arima_model_{region}_{period.replace('.', '').replace('-', '_')}.pkl",
    )
    save_model(model, model_path)

    # Evaluate model
    predictions, mae, rmse = evaluate_model(model, val_endog, val_exog)

    # Save predictions
    predictions_df = pd.DataFrame(
        {
            "DateTime": val_endog.index,
            "Actual": val_endog.values,
            "Predicted": predictions.values,
        }
    )
    predictions_df.to_csv(
        os.path.join(
            results_dir,
            f"predictions_{region}_{period.replace('.', '').replace('-', '_')}.csv",
        ),
        index=False,
    )

    # Plot results
    plot_path = os.path.join(
        results_dir, f"plot_{region}_{period.replace('.', '').replace('-', '_')}.png"
    )
    plot_results(
        val_endog, predictions, f"ARIMA Results for {region} {period}", plot_path
    )

    # Calculate average difference
    avg_diff = np.mean(np.abs(val_endog.values - predictions.values))

    # Print results
    print(f"\nResults for {region} {period}:")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Root Mean Squared Error: {rmse:.6f}")
    print(f"Average Difference: {avg_diff:.6f}")

    # Store results
    results = {
        "region": region,
        "period": period,
        "mae": mae,
        "rmse": rmse,
        "avg_diff": avg_diff,
    }

    # Save summary
    with open(os.path.join(results_dir, "results_summary.txt"), "w") as f:
        f.write(f"ARIMA Model Results for {region} {period}\n")
        f.write(f"Model Order: {arima_order}\n")
        f.write(f"Mean Absolute Error: {mae:.6f}\n")
        f.write(f"Root Mean Squared Error: {rmse:.6f}\n")
        f.write(f"Average Difference: {avg_diff:.6f}\n")

    return results


def run_all_experiments():
    """Run ARIMA experiments for all regions and periods"""
    regions = ["CN", "US"]
    periods = ["2023.01-2023.12", "2023.06-2024.06"]

    all_results = []

    for region in regions:
        for period in periods:
            results = run_arima_experiment(region, period)
            all_results.append(results)

    # Create overall summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("models/arima_summary_results.csv", index=False)

    print(
        "\nAll experiments completed. Summary saved to models/arima_summary_results.csv"
    )


if __name__ == "__main__":
    run_all_experiments()
