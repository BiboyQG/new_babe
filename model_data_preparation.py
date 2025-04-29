import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_dataset(file_path):
    """Load a dataset from a CSV file"""
    return pd.read_csv(file_path)


def prepare_for_arima(df, target_col="gap_close", datetime_col="DateTime"):
    """
    Prepare data for ARIMA model

    Returns:
    - time_series: pandas Series with DateTime index
    """
    # Make a copy of the dataframe
    df_copy = df.copy()

    # Convert datetime to index
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])
    df_copy = df_copy.set_index(datetime_col)

    # Extract the target column as a time series
    time_series = df_copy[target_col]

    return time_series


def prepare_for_xgboost(
    df, target_col="gap_close", datetime_col="DateTime", lag_periods=5
):
    """
    Prepare data for XGBoost model with lag features

    Returns:
    - X: feature matrix
    - y: target vector
    """
    # Make a copy of the dataframe
    df_copy = df.copy()

    # Convert datetime to index
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])
    df_copy = df_copy.set_index(datetime_col)

    # Create lag features for the target variable
    for i in range(1, lag_periods + 1):
        df_copy[f"{target_col}_lag_{i}"] = df_copy[target_col].shift(i)

    # Add time-based features
    df_copy["hour"] = df_copy.index.hour
    df_copy["minute"] = df_copy.index.minute
    df_copy["day_of_week"] = df_copy.index.dayofweek

    # Drop rows with NaN due to lag creation
    df_copy = df_copy.dropna()

    # Separate features and target
    X = df_copy.drop(target_col, axis=1)
    y = df_copy[target_col]

    return X, y


def create_sequences(data, seq_length, target_col="gap_close"):
    """
    Create sequences for sequence models (LSTM/GRU/Transformer)

    Returns:
    - X: input sequences (samples, timesteps, features)
    - y: target values
    """
    X, y = [], []

    # Get the column index of the target
    target_idx = data.columns.get_loc(target_col)

    for i in range(len(data) - seq_length):
        # Extract sequence of features
        seq = data.iloc[i : i + seq_length].values
        X.append(seq)

        # Extract target (next value of target column)
        target = data.iloc[i + seq_length, target_idx]
        y.append(target)

    return np.array(X), np.array(y)


def prepare_for_sequence_models(
    df, seq_length=24, target_col="gap_close", datetime_col="DateTime"
):
    """
    Prepare data for sequence models (LSTM/GRU/Transformer)

    Returns:
    - X: input sequences (samples, timesteps, features)
    - y: target values
    - scaler: the fitted scaler for inverse transform later
    """
    # Make a copy of the dataframe
    df_copy = df.copy()

    # Convert datetime to index
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])
    df_copy = df_copy.set_index(datetime_col)

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_copy.values)
    scaled_df = pd.DataFrame(scaled_data, columns=df_copy.columns, index=df_copy.index)

    # Create sequences
    X, y = create_sequences(scaled_df, seq_length, target_col)

    return X, y, scaler


def load_and_prepare_datasets(
    model_type, region, period, base_dir="prepared_data", seq_length=24, lag_periods=5
):
    """
    Load and prepare datasets for a specific model type

    Parameters:
    - model_type: 'arima', 'xgboost', 'lstm', 'transformer'
    - region: 'CN' or 'US'
    - period: '2023.01-2023.12' or '2023.06-2024.06'

    Returns:
    Depends on model_type:
    - arima: (train_ts, val_ts)
    - xgboost: (X_train, y_train, X_val, y_val)
    - lstm/transformer: (X_train, y_train, X_val, y_val, scaler)
    """
    # Define file paths
    filename_prefix = f"{region}_{period.replace('.', '').replace('-', '_')}"
    train_file = os.path.join(base_dir, f"{filename_prefix}_train.csv")
    val_file = os.path.join(base_dir, f"{filename_prefix}_val.csv")

    # Load datasets
    train_df = load_dataset(train_file)
    val_df = load_dataset(val_file)

    # Prepare based on model type
    if model_type.lower() == "arima":
        train_ts = prepare_for_arima(train_df)
        val_ts = prepare_for_arima(val_df)
        return train_ts, val_ts

    elif model_type.lower() == "xgboost":
        X_train, y_train = prepare_for_xgboost(train_df, lag_periods=lag_periods)
        X_val, y_val = prepare_for_xgboost(val_df, lag_periods=lag_periods)
        return X_train, y_train, X_val, y_val

    elif model_type.lower() in ["lstm", "gru", "transformer"]:
        X_train, y_train, scaler = prepare_for_sequence_models(
            train_df, seq_length=seq_length
        )
        X_val, y_val, _ = prepare_for_sequence_models(val_df, seq_length=seq_length)
        return X_train, y_train, X_val, y_val, scaler

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Example usage of the data preparation functions"""
    model_types = ["arima", "xgboost", "lstm", "transformer"]
    regions = ["CN", "US"]
    periods = ["2023.01-2023.12", "2023.06-2024.06"]

    # Example preparation for each model type
    for model_type in model_types:
        print(f"\nPreparing data for {model_type.upper()} models:")

        for region in regions:
            for period in periods:
                print(f"  {region} {period}...")

                if model_type == "arima":
                    train_ts, val_ts = load_and_prepare_datasets(
                        model_type, region, period
                    )
                    print(
                        f"    Train shape: {train_ts.shape}, Val shape: {val_ts.shape}"
                    )

                elif model_type == "xgboost":
                    X_train, y_train, X_val, y_val = load_and_prepare_datasets(
                        model_type, region, period
                    )
                    print(
                        f"    X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
                    )
                    print(f"    X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

                else:  # 'lstm' or 'transformer'
                    X_train, y_train, X_val, y_val, scaler = load_and_prepare_datasets(
                        model_type, region, period
                    )
                    print(
                        f"    X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
                    )
                    print(f"    X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")


if __name__ == "__main__":
    # Only run the example if script is called directly
    # main()
    pass
