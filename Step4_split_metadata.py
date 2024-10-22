import pandas as pd
import numpy as np

# Function to split the dataset
def split_dataset(input_file_path, eval_percentage, train_file_path, eval_file_path):

    train_df = pd.read_csv(input_file_path, delimiter="|")

    num_rows_to_move = int(len(train_df) * eval_percentage / 100)

    # Randomly sample rows
    rows_to_move = train_df.sample(n=num_rows_to_move, random_state=42)
    train_df = train_df.drop(rows_to_move.index)
    eval_df = rows_to_move

    train_df.to_csv(train_file_path, sep="|", index=False)
    eval_df.to_csv(eval_file_path, sep="|", index=False)

    print(f"Moved {num_rows_to_move} rows from {input_file_path} to {eval_file_path}")

# Terminal will talk to you.
def main():
    input_file_path = input("Enter the input .csv file path: ")
    eval_percentage = float(input("Enter the percentage of data to move to the evaluation set: "))
    train_file_path = input("Enter the path to save the training metadata file: ")
    eval_file_path = input("Enter the path to save the evaluation metadata file: ")
    split_dataset(input_file_path, eval_percentage, train_file_path, eval_file_path)

if __name__ == "__main__":
    main()
