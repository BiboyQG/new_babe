import os
import pandas as pd
import numpy as np
from datetime import datetime


def load_data(file_path, date_col="DateTime", encoding="utf-8"):
    """Load data files with appropriate encoding and date parsing"""
    print(f"Loading file: {file_path}")

    if file_path.endswith(".csv"):
        # Try different encodings if utf-8 fails
        try:
            df = pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding="gbk")
                print(f"  Used 'gbk' encoding for {file_path}")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin1")
                print(f"  Used 'latin1' encoding for {file_path}")
    else:  # Excel files
        df = pd.read_excel(file_path)

    # Print column names for debugging
    print(f"  Columns in {os.path.basename(file_path)}: {list(df.columns)}")

    # Special handling for inflation rate files
    if "inflation_rate" in file_path:
        # For US inflation files, fix column names
        if "us_inflation" in file_path:
            if (
                "Unnamed: 0" in df.columns
                and "PCEPI：2017年价格：SA：环比百分比" in df.columns
            ):
                df = df.rename(
                    columns={
                        "Unnamed: 0": "date",
                        "PCEPI：2017年价格：SA：环比百分比": "index",
                    }
                )
                print(f"  Renamed columns for US inflation file: {list(df.columns)}")

    # Convert date column to datetime if it exists
    if date_col in df.columns:
        try:
            # Try different formats depending on the file type
            if "interest_rate" in file_path:
                # Handle special date formats in interest rate files
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            elif "oil_price" in file_path:
                if "us_oil" in file_path:
                    # US oil prices might use MM/DD/YYYY format
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                    print(
                        f"  Date format in {os.path.basename(file_path)}: {df[date_col].iloc[0]}"
                    )
                else:
                    # Try multiple formats for oil price files
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            else:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

            # Check if conversion was successful
            if df[date_col].isna().all():
                print(
                    f"Warning: Could not convert {date_col} to datetime in {file_path}"
                )
        except Exception as e:
            print(
                f"Warning: Could not convert {date_col} to datetime in {file_path}: {str(e)}"
            )

    # Handle US oil price date formats specifically
    if "us_oil_price" in file_path and not df[date_col].dtype == "datetime64[ns]":
        try:
            # First check format in the data
            sample_date = df[date_col].iloc[0]
            print(f"  Sample date in {os.path.basename(file_path)}: {sample_date}")

            if "/" in str(sample_date):
                # Try M/D/YYYY format first (US format)
                df[date_col] = pd.to_datetime(
                    df[date_col], format="%m/%d/%Y", errors="coerce"
                )
            else:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

            print(
                f"  Converted dates in {os.path.basename(file_path)}: {df[date_col].iloc[0]}"
            )
        except Exception as e:
            print(
                f"  Error converting dates in {os.path.basename(file_path)}: {str(e)}"
            )

    return df


def merge_features_with_target(
    target_df, feature_dfs, feature_names, date_col="DateTime"
):
    """Merge all feature dataframes with the target dataframe based on date"""
    # Make a copy of the target dataframe
    result_df = target_df.copy()

    # Ensure the date column is the index and is datetime type
    if date_col in result_df.columns:
        result_df[date_col] = pd.to_datetime(result_df[date_col], errors="coerce")
        result_df = result_df.set_index(date_col)

    # Merge each feature dataframe
    for i, (feature_df, feature_name) in enumerate(zip(feature_dfs, feature_names)):
        print(f"Processing feature: {feature_name}")

        # Handle different date column names
        date_column = None
        if date_col in feature_df.columns:
            date_column = date_col
        elif "date" in feature_df.columns:
            date_column = "date"
        elif "Effective Date" in feature_df.columns:
            date_column = "Effective Date"

        if date_column:
            print(f"  Using date column: {date_column}")
            # Ensure it's datetime type
            feature_df[date_column] = pd.to_datetime(
                feature_df[date_column], errors="coerce"
            )
            # Report on conversion success
            na_count = feature_df[date_column].isna().sum()
            if na_count > 0:
                print(
                    f"  Warning: {na_count} dates could not be parsed in {feature_name}"
                )
            # Drop rows where datetime conversion failed
            feature_df = feature_df.dropna(subset=[date_column])
            feature_df = feature_df.set_index(date_column)
        else:
            print(f"  Warning: No date column found in {feature_name}")
            continue

        # Check if the index is a DatetimeIndex
        if not isinstance(feature_df.index, pd.DatetimeIndex):
            print(
                f"  Warning: Skipping {feature_name} as it doesn't have a valid datetime index"
            )
            continue

        # Identify numeric columns to merge
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        print(f"  Numeric columns in {feature_name}: {list(numeric_cols)}")

        # For each numeric column, resample to match target_df's frequency
        for col in numeric_cols:
            try:
                # Create a Series with the feature
                feature_series = feature_df[col]
                print(f"  Processing column: {col}, Values: {feature_series.head(3)}")

                # Forward fill to handle the minute-level granularity
                resampled_feature = feature_series.resample("1min").ffill()

                # Check if resampling worked
                if resampled_feature.isna().all():
                    print(
                        f"  Warning: All values are NaN after resampling {col} from {feature_name}"
                    )
                    continue

                # Add the feature to the result dataframe, with a prefix to identify the source
                feature_col_name = f"{feature_name}_{col}"
                before_join_shape = result_df.shape
                result_df = result_df.join(
                    resampled_feature.rename(feature_col_name), how="left"
                )

                # Check if join was successful
                if feature_col_name in result_df.columns:
                    non_null_count = result_df[feature_col_name].count()
                    print(
                        f"  Successfully joined {feature_col_name}, non-null values: {non_null_count}"
                    )
                else:
                    print(f"  Warning: Failed to join {feature_col_name}")
            except Exception as e:
                print(f"  Error resampling {col} from {feature_name}: {str(e)}")

    # Forward fill any missing values from features
    result_df = result_df.fillna(method="ffill")
    # Then backward fill to handle start of data
    result_df = result_df.fillna(method="bfill")

    # Check for remaining NaN values
    nan_cols = result_df.columns[result_df.isna().any()].tolist()
    if nan_cols:
        print(f"Columns with NaN values after fill: {nan_cols}")
        # Fill remaining NAs with 0
        result_df = result_df.fillna(0)

    # Reset index to have DateTime as a column again
    result_df = result_df.reset_index()

    # Final check for all columns
    print(f"Final dataset columns: {list(result_df.columns)}")
    # Check values for key columns
    for feature_name in feature_names:
        for col in result_df.columns:
            if feature_name in col:
                sample_values = result_df[col].head(3).tolist()
                print(f"  Sample values for {col}: {sample_values}")
                zero_count = (result_df[col] == 0).sum()
                if zero_count > 0:
                    print(
                        f"  Warning: {zero_count} zero values in {col} ({zero_count / len(result_df) * 100:.2f}%)"
                    )

    return result_df


def train_val_split(df, val_ratio=0.2, time_based=True):
    """Split data into training and validation sets"""
    if time_based:
        # Time-based split (last val_ratio of data for validation)
        split_idx = int(len(df) * (1 - val_ratio))
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
    else:
        # Random split
        train_df = df.sample(frac=(1 - val_ratio), random_state=42)
        val_df = df.drop(train_df.index)

    return train_df, val_df


def prepare_dataset(region, period, val_ratio=0.2, output_dir="prepared_data"):
    """
    Prepare a dataset for a specific region and period

    Parameters:
    region: 'CN' or 'US'
    period: '2023.01-2023.12' or '2023.06-2024.06'
    """
    print(f"\nPreparing dataset for {region} {period}...")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine target and feature files based on region and period
    if region == "CN":
        if period == "2023.01-2023.12":
            target_file = "data/cn_price_gaps_AU2312.csv"
            feature_files = [
                "data/cn_inflation_rate_calculated_2023.01.01-2023.12.01.xlsx",
                "data/cn_interest_rate_DR007_2023.01.01-2023.12.31.csv",
                "data/cn_oil_price_SC00.INE_2023.01.01-2023.12.31.csv",
            ]
            feature_names = ["inflation", "interest", "oil"]
        else:  # 2023.06-2024.06
            target_file = "data/cn_price_gaps_AU2406.csv"
            feature_files = [
                "data/cn_inflation_rate_calculated_2023.06.01-2024.06.01.xlsx",
                "data/cn_interest_rate_DR007_2023.06.01-2024.06.01.csv",
                "data/cn_oil_price_SC00.INE_2023.06.01-2024.06.01.csv",
            ]
            feature_names = ["inflation", "interest", "oil"]
    else:  # US
        if period == "2023.01-2023.12":
            target_file = "data/us_price_gaps_GCZ23E.CMX.csv"
            feature_files = [
                "data/us_inflation_rate_calculated_2023.01.01-2023.12.31.xlsx",
                "data/us_interest_rate_GCZ23E_real_ffr.xlsx",
                "data/us_oil_price_CL00.NYM_2023.01.01-2023.12.31.csv",
            ]
            feature_names = ["inflation", "interest", "oil"]
        else:  # 2023.06-2024.06
            target_file = "data/us_price_gaps_GCM24E.CMX.csv"
            feature_files = [
                "data/us_inflation_rate_calculated_2023.06.01-2024.06.01.xlsx",
                "data/us_interest_rate_GCM24E_real_ffr.xlsx",
                "data/us_oil_price_CL00.NYM_2023.06.01-2024.06.01.csv",
            ]
            feature_names = ["inflation", "interest", "oil"]

    # Load target data
    print(f"Loading target data from {target_file}...")
    target_df = load_data(target_file)

    # Load feature data
    feature_dfs = []
    for feature_file in feature_files:
        print(f"Loading feature data from {feature_file}...")
        feature_dfs.append(load_data(feature_file))

    # Merge features with target
    print("Merging features with target...")
    merged_df = merge_features_with_target(target_df, feature_dfs, feature_names)

    # Split data into training and validation sets
    print("Splitting data into training and validation sets...")
    train_df, val_df = train_val_split(merged_df, val_ratio=val_ratio)

    # Save the datasets
    output_prefix = f"{region}_{period.replace('.', '').replace('-', '_')}"
    train_file = os.path.join(output_dir, f"{output_prefix}_train.csv")
    val_file = os.path.join(output_dir, f"{output_prefix}_val.csv")

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)

    print(f"Training set saved to {train_file} ({len(train_df)} rows)")
    print(f"Validation set saved to {val_file} ({len(val_df)} rows)")

    return train_df, val_df


def main():
    # Create prepared data directory
    output_dir = "prepared_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare all datasets
    datasets = [
        ("CN", "2023.01-2023.12"),
        ("CN", "2023.06-2024.06"),
        ("US", "2023.01-2023.12"),
        ("US", "2023.06-2024.06"),
    ]

    for region, period in datasets:
        prepare_dataset(region, period, output_dir=output_dir)

    print("\nAll datasets have been prepared successfully!")


if __name__ == "__main__":
    main()
