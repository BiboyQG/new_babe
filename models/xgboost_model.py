import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_dataset(file_path):
    """Load a dataset from a CSV file"""
    return pd.read_csv(file_path)


def prepare_data_for_xgboost(
    df, target_col="gap_open", datetime_col="DateTime", feature_cols=None
):
    """
    Prepare data for XGBoost model

    Parameters:
    - df: DataFrame containing the data
    - target_col: Target column to predict
    - datetime_col: Column containing datetime information
    - feature_cols: List of feature columns to include

    Returns:
    - X: Feature DataFrame
    - y: Target Series
    - datetime_index: pandas DatetimeIndex
    """
    # Make a copy of the dataframe
    df_copy = df.copy()

    # Convert datetime to index
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])
    datetime_index = df_copy[datetime_col]
    df_copy = df_copy.set_index(datetime_col)

    # Extract the target column
    y = df_copy[target_col]

    # Extract features
    X = None
    if feature_cols:
        X = df_copy[feature_cols]
    else:
        # Use all columns except the target and datetime
        X = df_copy.drop([target_col], axis=1)

    return X, y, datetime_index


def train_xgboost_model(X_train, y_train, params=None):
    """
    Train XGBoost model

    Parameters:
    - X_train: Training features
    - y_train: Training target
    - params: XGBoost parameters (optional)

    Returns:
    - model: Trained XGBoost model
    """
    # Default parameters if none provided
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "learning_rate": 0.1,
            "max_depth": 5,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 100,
        }

    print("Training XGBoost model...")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    print("Model training complete.")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate XGBoost model on test data

    Parameters:
    - model: Trained XGBoost model
    - X_test: Test features
    - y_test: Test target

    Returns:
    - predictions: Predicted values
    - mae: Mean Absolute Error
    - rmse: Root Mean Squared Error
    - avg_diff: Average absolute difference between predictions and actuals
    """
    # Generate predictions
    predictions = model.predict(X_test)

    # Calculate error metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    avg_diff = np.mean(np.abs(y_test.values - predictions))

    return predictions, mae, rmse, avg_diff


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


def plot_results(actual, predicted, datetime_index, title, save_path=None):
    """
    Plot actual vs predicted values

    Parameters:
    - actual: Actual values (Series)
    - predicted: Predicted values (array)
    - datetime_index: DatetimeIndex for the x-axis
    - title: Plot title
    - save_path: Path to save the plot (optional)
    """
    # Create a DataFrame with both actual and predicted values
    results_df = pd.DataFrame(
        {"DateTime": datetime_index, "Actual": actual.values, "Predicted": predicted}
    )
    results_df = results_df.set_index("DateTime")

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


def run_xgboost_experiment(
    region, period, base_dir="prepared_data", models_dir="models"
):
    """
    Run complete XGBoost experiment for a given region and period

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
        models_dir, "xgboost", region, period.replace(".", "").replace("-", "_")
    )
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nRunning XGBoost experiment for {region} {period}...")

    # Load datasets
    train_df = load_dataset(train_file)
    val_df = load_dataset(val_file)

    # Determine feature column names based on region
    if region == "CN":
        feature_cols = ["inflation_index", "interest_OPEN", "oil_CLOSE"]
    else:
        feature_cols = ["inflation_index", "interest_Rate (%)", "oil_CLOSE"]

    # Prepare data
    X_train, y_train, train_datetime = prepare_data_for_xgboost(
        train_df, target_col="gap_open", feature_cols=feature_cols
    )
    X_val, y_val, val_datetime = prepare_data_for_xgboost(
        val_df, target_col="gap_open", feature_cols=feature_cols
    )

    # Define model parameters
    xgb_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 200,
        "random_state": 42,
    }

    # Train model
    model = train_xgboost_model(X_train, y_train, params=xgb_params)

    # Save model
    model_path = os.path.join(
        results_dir,
        f"xgboost_model_{region}_{period.replace('.', '').replace('-', '_')}.pkl",
    )
    save_model(model, model_path)

    # Evaluate model
    predictions, mae, rmse, avg_diff = evaluate_model(model, X_val, y_val)

    # Save predictions
    predictions_df = pd.DataFrame(
        {
            "DateTime": val_datetime,
            "Actual": y_val.values,
            "Predicted": predictions,
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
        y_val,
        predictions,
        val_datetime,
        f"XGBoost Results for {region} {period}",
        plot_path,
    )

    # Print results
    print(f"\nResults for {region} {period}:")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Root Mean Squared Error: {rmse:.6f}")
    print(f"Average Difference: {avg_diff:.6f}")

    # Feature importance
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_cols, "Importance": feature_importance}
    ).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:")
    for index, row in feature_importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.6f}")

    feature_importance_df.to_csv(
        os.path.join(
            results_dir,
            f"feature_importance_{region}_{period.replace('.', '').replace('-', '_')}.csv",
        ),
        index=False,
    )

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
        f.write(f"XGBoost Model Results for {region} {period}\n")
        f.write(f"Parameters: {xgb_params}\n")
        f.write(f"Mean Absolute Error: {mae:.6f}\n")
        f.write(f"Root Mean Squared Error: {rmse:.6f}\n")
        f.write(f"Average Difference: {avg_diff:.6f}\n")
        f.write("\nFeature Importance:\n")
        for index, row in feature_importance_df.iterrows():
            f.write(f"{row['Feature']}: {row['Importance']:.6f}\n")

    return results


def run_all_experiments():
    """Run XGBoost experiments for all regions and periods"""
    regions = ["CN", "US"]
    periods = ["2023.01-2023.12", "2023.06-2024.06"]

    all_results = []

    for region in regions:
        for period in periods:
            results = run_xgboost_experiment(region, period)
            all_results.append(results)

    # Create overall summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("models/xgboost_summary_results.csv", index=False)

    print(
        "\nAll experiments completed. Summary saved to models/xgboost_summary_results.csv"
    )


if __name__ == "__main__":
    run_all_experiments()
