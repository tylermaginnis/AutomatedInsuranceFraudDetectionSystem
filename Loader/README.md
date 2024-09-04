# Loader

## Overview

The `Loader` module is responsible for loading, cleaning, preprocessing, and saving insurance claims data. This module ensures that the data is in a suitable format for further analysis and fraud detection.

## Components

### Loader.py

The `Loader.py` script is the core of the `Loader` module. It handles the entire data preparation pipeline, from loading raw data to saving the cleaned and normalized data.

#### Key Functions

- `load_data(data_dir)`: Loads data from JSON files in the specified directory and returns a DataFrame containing the loaded data.
- `clean_and_preprocess_data(df)`: Cleans and preprocesses the data, handling missing values, normalizing numeric features, and creating new features like claim frequency.
- `save_data(df, output_file)`: Saves the cleaned and preprocessed data to a JSON file.
- `main()`: The main function that parses command-line arguments and orchestrates the loading, cleaning, preprocessing, and saving of data.

## Usage

To load, clean, preprocess, and save insurance claims data, run the `Loader.py` script with the desired parameters:
