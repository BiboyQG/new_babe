import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from tqdm import tqdm


def load_dataset(file_path):
    """Load a dataset from a CSV file"""
    return pd.read_csv(file_path)


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        return out


def get_device():
    """
    Determine the best available device: MPS (for MacOS), CUDA, or CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def prepare_data_for_lstm(
    df,
    target_col="gap_open",
    datetime_col="DateTime",
    feature_cols=None,
    seq_length=60,
    train_ratio=1.0,
):
    """
    Prepare data for LSTM model

    Parameters:
    - df: DataFrame containing the data
    - target_col: Target column to predict
    - datetime_col: Column containing datetime information
    - feature_cols: List of feature columns to include
    - seq_length: Length of input sequences
    - train_ratio: Ratio of data to use for training (1.0 for full dataset)

    Returns:
    - X_tensor: Input tensor of shape (n_samples, seq_length, n_features)
    - y_tensor: Target tensor of shape (n_samples, 1)
    - datetime_index: Datetime indices for the samples
    - scalers: Dictionary containing scalers for features and target
    """
    # Make a copy of the dataframe
    df_copy = df.copy()

    # Convert datetime to index
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])
    datetime_index = df_copy[datetime_col]
    df_copy = df_copy.set_index(datetime_col)

    # Determine features to use
    if feature_cols is None:
        # Use all columns except the target
        feature_cols = [col for col in df_copy.columns if col != target_col]

    # Create feature and target DataFrames
    features_df = df_copy[feature_cols]
    target_df = df_copy[[target_col]]

    # Scale the data
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(features_df)
    scaled_target = target_scaler.fit_transform(target_df)

    # Prepare sequences
    print(f"Creating sequences with length {seq_length}...")
    X, y, dates = [], [], []

    # Use tqdm for the sequence creation process
    for i in tqdm(range(len(scaled_features) - seq_length), desc="Creating sequences"):
        X.append(scaled_features[i : i + seq_length])
        y.append(scaled_target[i + seq_length])
        dates.append(datetime_index.iloc[i + seq_length])

    # Convert to numpy arrays
    X_array = np.array(X)
    y_array = np.array(y)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_array)
    y_tensor = torch.FloatTensor(y_array)

    # Store the scalers for inverse transformation
    scalers = {"feature_scaler": feature_scaler, "target_scaler": target_scaler}

    return X_tensor, y_tensor, dates, scalers


def train_lstm_model(train_loader, model, epochs=100, learning_rate=0.001, device=None):
    """
    Train LSTM model

    Parameters:
    - train_loader: DataLoader containing training data
    - model: LSTM model to train
    - epochs: Number of training epochs
    - learning_rate: Learning rate for optimizer
    - device: PyTorch device ('mps', 'cuda' or 'cpu')

    Returns:
    - model: Trained model
    - losses: List of training losses
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    print(f"Training LSTM model on {device} for {epochs} epochs...")

    # Main training loop with progress bar for epochs
    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        model.train()
        train_loss = 0

        # Inner loop with progress bar for batches (only displayed for the first epoch)
        batch_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
            unit="batch",
            disable=epoch > 0,  # Only show for first epoch
        )

        for X_batch, y_batch in batch_progress:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Update batch progress bar with current loss
            batch_progress.set_postfix(loss=f"{loss.item():.6f}")

        # Calculate average loss for the epoch
        avg_loss = train_loss / len(train_loader)
        losses.append(avg_loss)

        # Print progress (less frequently with tqdm showing progress)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

    print("Model training complete.")
    return model, losses


def evaluate_model(model, test_loader, scaler, device=None):
    """
    Evaluate LSTM model on test data

    Parameters:
    - model: Trained LSTM model
    - test_loader: DataLoader containing test data
    - scaler: Scaler to inverse transform predictions
    - device: PyTorch device ('mps', 'cuda' or 'cpu')

    Returns:
    - predictions: Numpy array of predictions (inverse transformed)
    - actuals: Numpy array of actual values (inverse transformed)
    - mae: Mean Absolute Error
    - rmse: Root Mean Squared Error
    - avg_diff: Average absolute difference
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    predictions = []
    actuals = []

    print("Evaluating model on validation data...")
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            X_batch = X_batch.to(device)

            # Forward pass
            outputs = model(X_batch)

            # Move tensors to CPU and convert to numpy
            pred_np = outputs.cpu().numpy()
            actual_np = y_batch.cpu().numpy()

            # Store predictions and actuals
            predictions.append(pred_np)
            actuals.append(actual_np)

    # Concatenate batches
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    # Inverse transform
    predictions_rescaled = scaler.inverse_transform(predictions)
    actuals_rescaled = scaler.inverse_transform(actuals)

    # Calculate metrics
    mae = mean_absolute_error(actuals_rescaled, predictions_rescaled)
    rmse = np.sqrt(mean_squared_error(actuals_rescaled, predictions_rescaled))
    avg_diff = np.mean(np.abs(actuals_rescaled - predictions_rescaled))

    return predictions_rescaled, actuals_rescaled, mae, rmse, avg_diff


def save_model(model, scalers, file_path):
    """
    Save the trained model and scalers to disk

    Parameters:
    - model: Trained PyTorch model
    - scalers: Dictionary containing scalers for features and target
    - file_path: Path to save the model
    """
    # Create a dictionary to store both model and scalers
    save_dict = {
        "model_state_dict": model.state_dict(),
        "feature_scaler": scalers["feature_scaler"],
        "target_scaler": scalers["target_scaler"],
    }

    with open(file_path, "wb") as f:
        torch.save(save_dict, f)

    print(f"Model saved to {file_path}")


def load_model(file_path, input_size, hidden_size, num_layers, output_size):
    """
    Load a saved model from disk

    Parameters:
    - file_path: Path to the saved model
    - input_size: Number of input features
    - hidden_size: Size of hidden layers
    - num_layers: Number of LSTM layers
    - output_size: Size of output layer

    Returns:
    - model: Loaded PyTorch model
    - scalers: Dictionary containing scalers for features and target
    """
    # Initialize model with same architecture
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # Load saved model and scalers
    save_dict = torch.load(file_path, map_location=torch.device("cpu"))
    model.load_state_dict(save_dict["model_state_dict"])

    scalers = {
        "feature_scaler": save_dict["feature_scaler"],
        "target_scaler": save_dict["target_scaler"],
    }

    return model, scalers


def plot_results(actual, predicted, dates, title, save_path=None):
    """
    Plot actual vs predicted values

    Parameters:
    - actual: Actual values (array)
    - predicted: Predicted values (array)
    - dates: Datetime indices for the x-axis
    - title: Plot title
    - save_path: Path to save the plot (optional)
    """
    # Create a DataFrame with both actual and predicted values
    results_df = pd.DataFrame(
        {
            "DateTime": dates,
            "Actual": actual.flatten(),
            "Predicted": predicted.flatten(),
        }
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


def run_lstm_experiment(region, period, base_dir="prepared_data", models_dir="models"):
    """
    Run complete LSTM experiment for a given region and period

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
        models_dir, "lstm", region, period.replace(".", "").replace("-", "_")
    )
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nRunning LSTM experiment for {region} {period}...")

    # Load datasets
    train_df = load_dataset(train_file)
    val_df = load_dataset(val_file)

    # Determine feature column names based on region
    if region == "CN":
        feature_cols = ["inflation_index", "interest_OPEN", "oil_CLOSE"]
    else:
        feature_cols = ["inflation_index", "interest_Rate (%)", "oil_CLOSE"]

    # Set model hyperparameters
    seq_length = 60  # Length of input sequences (can be tuned)
    batch_size = 64
    hidden_size = 64
    num_layers = 2
    learning_rate = 0.001
    epochs = 100

    # Prepare data for LSTM
    X_train, y_train, train_dates, scalers = prepare_data_for_lstm(
        train_df,
        target_col="gap_open",
        feature_cols=feature_cols,
        seq_length=seq_length,
    )

    X_val, y_val, val_dates, _ = prepare_data_for_lstm(
        val_df, target_col="gap_open", feature_cols=feature_cols, seq_length=seq_length
    )

    # Create DataLoaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_size = len(feature_cols)  # Number of features
    output_size = 1  # We're predicting a single value (gap_open)

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # Train model
    model, losses = train_lstm_model(
        train_loader, model, epochs=epochs, learning_rate=learning_rate, device=device
    )

    # Save model
    model_path = os.path.join(
        results_dir,
        f"lstm_model_{region}_{period.replace('.', '').replace('-', '_')}.pt",
    )
    save_model(model, scalers, model_path)

    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(f"Training Loss for {region} {period}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(
        os.path.join(
            results_dir,
            f"training_loss_{region}_{period.replace('.', '').replace('-', '_')}.png",
        )
    )
    plt.close()

    # Evaluate model
    predictions, actuals, mae, rmse, avg_diff = evaluate_model(
        model, val_loader, scalers["target_scaler"], device=device
    )

    # Save predictions
    predictions_df = pd.DataFrame(
        {
            "DateTime": val_dates[: len(predictions)],
            "Actual": actuals.flatten(),
            "Predicted": predictions.flatten(),
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
        actuals,
        predictions,
        val_dates[: len(predictions)],
        f"LSTM Results for {region} {period}",
        plot_path,
    )

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

    # Save hyperparameters and results
    with open(os.path.join(results_dir, "results_summary.txt"), "w") as f:
        f.write(f"LSTM Model Results for {region} {period}\n")
        f.write(f"Sequence Length: {seq_length}\n")
        f.write(f"Hidden Size: {hidden_size}\n")
        f.write(f"Number of Layers: {num_layers}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Mean Absolute Error: {mae:.6f}\n")
        f.write(f"Root Mean Squared Error: {rmse:.6f}\n")
        f.write(f"Average Difference: {avg_diff:.6f}\n")

    return results


def run_all_experiments():
    """Run LSTM experiments for all regions and periods"""
    regions = ["CN", "US"]
    periods = ["2023.01-2023.12", "2023.06-2024.06"]

    # Check for MPS availability and print info
    if torch.backends.mps.is_available():
        print(
            "MPS (Metal Performance Shaders) is available! Using GPU acceleration on macOS."
        )
    elif torch.cuda.is_available():
        print("CUDA is available! Using GPU acceleration.")
    else:
        print("No GPU acceleration available. Using CPU for computations.")

    all_results = []

    for region in regions:
        for period in periods:
            results = run_lstm_experiment(region, period)
            all_results.append(results)

    # Create overall summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("models/lstm_summary_results.csv", index=False)

    print(
        "\nAll experiments completed. Summary saved to models/lstm_summary_results.csv"
    )


if __name__ == "__main__":
    run_all_experiments()
