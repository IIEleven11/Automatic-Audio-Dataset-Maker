import os
import glob
import pandas as pd
from pathlib import Path

project_root = os.path.dirname(os.path.abspath(__file__))
UNFILTERED_PARQUET_DIR = os.path.join(project_root, "UNFILTERED_PARQUET")
os.makedirs(UNFILTERED_PARQUET_DIR, exist_ok=True)

input_dir = os.path.join(project_root, "UNFILTERED_PARQUET")
output_dir = os.path.join(project_root, "FILTERED_PARQUET")

pesq_threshold = 3.5
snr_threshold = 30
stoi_threshold = 0.95
c50_threshold = 55
si_sdr_threshold = 15


parquet_files = glob.glob(os.path.join(input_dir, '*.parquet'))


if not parquet_files:
    print(f"No Parquet files found in {input_dir}")
else:
    print(f"Found {len(parquet_files)} Parquet files in {input_dir}")


for parquet_file in parquet_files:
    print(f"Processing file: {parquet_file}")
    

    df = pd.read_parquet(parquet_file)
    

    print(f"First few rows of {parquet_file}:")
    print(df.head())

    # Apply the filters
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


    base_name = os.path.basename(parquet_file)
    output_file = os.path.join(output_dir, base_name.replace('.parquet', '_filtered.parquet'))


    df_filtered.to_parquet(output_file, engine='pyarrow')

    print(f"Filtered data saved to {output_file}")
