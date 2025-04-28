import os
import pandas as pd


def analyze_data_files():
    data_folder = "data"
    output_file = "data_preview.txt"

    # Get list of all CSV and XLSX files in the data folder
    files = [f for f in os.listdir(data_folder) if f.endswith((".csv", ".xlsx"))]
    files.sort()  # Sort files alphabetically

    with open(output_file, "w") as f:
        f.write("# Data Files Analysis\n\n")
        f.write(
            "This file contains a preview of all CSV and XLSX files in the data folder.\n\n"
        )

        for file in files:
            file_path = os.path.join(data_folder, file)
            f.write(f"\n{'=' * 80}\n")
            f.write(f"## File: {file}\n")
            f.write(f"{'=' * 80}\n\n")

            try:
                # Read the file based on its extension
                if file.endswith(".csv"):
                    # Try different encodings for CSV files
                    encodings = ["utf-8", "gbk", "latin1", "ISO-8859-1", "cp1252"]
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding)
                            f.write(
                                f"File successfully read with encoding: {encoding}\n"
                            )
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        # If all encodings fail
                        raise UnicodeDecodeError("All encodings failed", "", 0, 0, "")
                else:  # xlsx
                    df = pd.read_excel(file_path)

                # Write file information
                f.write(f"### Basic Information\n")
                f.write(f"- File size: {os.path.getsize(file_path) / 1024:.2f} KB\n")
                f.write(f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n")

                # Write column information
                f.write(f"### Columns\n")
                for col in df.columns:
                    f.write(f"- **{col}**: {df[col].dtype}\n")
                f.write("\n")

                # Write descriptive statistics if numerical columns exist
                num_cols = df.select_dtypes(include=["number"]).columns
                if len(num_cols) > 0:
                    f.write(f"### Descriptive Statistics for Numerical Columns\n")
                    f.write(df[num_cols].describe().to_string())
                    f.write("\n\n")

                # Write first 5 rows
                f.write("### First 5 Rows\n")
                f.write(df.head().to_string())
                f.write("\n\n")

                # Note time range if datetime column exists
                date_cols = [
                    col
                    for col in df.columns
                    if any(
                        date_term in col.lower()
                        for date_term in ["date", "time", "dt", "day"]
                    )
                ]
                if date_cols:
                    f.write("### Date Range\n")
                    for date_col in date_cols:
                        try:
                            if df[date_col].dtype == "object":
                                df[date_col] = pd.to_datetime(
                                    df[date_col], errors="coerce"
                                )
                            min_date = df[date_col].min()
                            max_date = df[date_col].max()
                            f.write(f"- **{date_col}**: {min_date} to {max_date}\n")
                        except:
                            f.write(f"- **{date_col}**: Could not parse as date\n")
                    f.write("\n")

            except Exception as e:
                f.write(f"Error reading file: {str(e)}\n\n")

    print(f"Analysis completed. Results saved to {output_file}")


if __name__ == "__main__":
    analyze_data_files()
