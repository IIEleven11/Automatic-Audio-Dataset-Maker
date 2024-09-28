import os
import glob
import pandas as pd

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    UNFILTERED_PARQUET_DIR = os.path.join(project_root, "UNFILTERED_PARQUET")
    FILTERED_PARQUET_DIR = os.path.join(project_root, "FILTERED_PARQUET")
    os.makedirs(FILTERED_PARQUET_DIR, exist_ok=True)

    # Define filtering thresholds
    pesq_threshold = 3.5
    snr_threshold = 30
    stoi_threshold = 0.95
    c50_threshold = 55
    si_sdr_threshold = 15

    # Find all Parquet files in the converted directory
    parquet_files = glob.glob(os.path.join(UNFILTERED_PARQUET_DIR, '**', '*.parquet'), recursive=True)

    if not parquet_files:
        print(f"No Parquet files found in {UNFILTERED_PARQUET_DIR}")
        return
    print(f"Found {len(parquet_files)} Parquet files in {UNFILTERED_PARQUET_DIR}")

    for parquet_file in parquet_files:
        print(f"Processing file: {parquet_file}")
        
        df = pd.read_parquet(parquet_file)
        print(f"First few rows of {parquet_file}:")
        print(df.head())
        df_filtered = df[
            (df['pesq'] > pesq_threshold) &
            (df['snr'] > snr_threshold) &
            (df['stoi'] > stoi_threshold) &
            (df['c50'] > c50_threshold) &
            (df['si-sdr'] > si_sdr_threshold)
        ]
        if df_filtered.empty:
            print(f"No rows passed the filters for {parquet_file}")
        else:
            print(f"{len(df_filtered)} rows passed the filters for {parquet_file}")

        relative_path = os.path.relpath(parquet_file, UNFILTERED_PARQUET_DIR)
        output_file = os.path.join(FILTERED_PARQUET_DIR, relative_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_filtered.to_parquet(output_file, engine='pyarrow')

        print(f"Filtered data saved to {output_file}")

if __name__ == "__main__":
    main()
