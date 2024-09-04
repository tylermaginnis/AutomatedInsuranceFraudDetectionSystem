import os
import json
import pandas as pd
from glob import glob
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_dir):
    """
    Load data from JSON files in the specified directory.

    Args:
        data_dir (str): The directory containing JSON files.

    Returns:
        list: List containing the loaded data.
    """
    files = glob(os.path.join(data_dir, "*.json"))
    data = []
    for file in files:
        try:
            with open(file, 'r') as f:
                data.append(json.load(f))
            logging.info(f"Loaded data from {file}")
        except Exception as e:
            logging.error(f"Error loading data from {file}: {e}")
    return data

def clean_and_preprocess_data(data):
    """
    Clean and preprocess the data.

    Args:
        data (list): The input data.

    Returns:
        list: The cleaned and preprocessed data.
    """
    df = pd.DataFrame(data)
    
    # Handle missing values
    df.fillna(0, inplace=True)
    logging.info("Handled missing values by filling with 0")
    
    # Normalize features
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
    logging.info("Normalized numeric features")
    
    # Create new features like claim frequency
    if 'PolicyHolderID' in df.columns and 'ClaimID' in df.columns:
        df['ClaimFrequency'] = df.groupby('PolicyHolderID')['ClaimID'].transform('count')
        logging.info("Created new feature 'ClaimFrequency'")
    else:
        logging.warning("Required columns 'PolicyHolderID' or 'ClaimID' not found in DataFrame")
    
    return df.to_dict(orient='records')

def save_data(data, output_file):
    """
    Save the cleaned and preprocessed data to a JSON file.

    Args:
        data (list): The data to save.
        output_file (str): The file path to save the data.
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Saved cleaned data to {output_file}")
    except Exception as e:
        logging.error(f"Error saving data to {output_file}: {e}")

def main():
    """
    Main function to load, clean, preprocess, and save data.
    """
    parser = argparse.ArgumentParser(description="Data Cleaning and Preparation")
    parser.add_argument("-d", "--data_dir", type=str, default="Simulation/Data", help="Directory containing JSON data files")
    parser.add_argument("-o", "--output_file", type=str, default="Loader/Data/normalized.json", help="Output file for cleaned data")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_file = args.output_file
    data = load_data(data_dir)
    if data:
        data = clean_and_preprocess_data(data)
        save_data(data, output_file)
    else:
        logging.error("No data loaded. Exiting.")

if __name__ == "__main__":
    main()
