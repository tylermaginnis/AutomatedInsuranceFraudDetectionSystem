# MLTool

## Overview

The `MLTool` module is designed to detect fraudulent insurance claims using machine learning models. It provides functionalities to train a fraud detection model, execute the model on normalized claim data, and generate SHAP values for model interpretability.

## Components

### MLTool.py

The `MLTool.py` script is the main component of the `MLTool` module. It includes functions to train the fraud detection model, process normalized claim data, and generate SHAP values.

#### Key Functions

- `train_model(training_data, model_path, generate_shap=False)`: Trains the fraud detection model using the provided training data and saves the trained model to the specified path. Optionally generates SHAP values for model interpretability.
- `process_normalized_json(input_file, output_dir, model_path)`: Processes the normalized claim data from the input file, calculates fraud likelihood for each claim using the trained model, and saves the results to the output directory.
- `calculate_fraud_likelihood(model, claim)`: Calculates the fraud likelihood for a given claim using the trained model.
- `extract_features_from_claim(claim)`: Extracts features from a claim to be used as input for the fraud detection model.
- `load_model(model_path)`: Loads the trained fraud detection model from the specified path.
- `main()`: The main function that handles command-line arguments and coordinates the training and execution of the model.

## Usage

To train the fraud detection model, execute the `MLTool.py` script with the `-t` flag:
- `-t` or `--train`: Trains the fraud detection model using the provided training data.

To generate SHAP values for model interpretability, use the `-s` flag:
- `-s` or `--shap`: Generates SHAP values without checking additivity for model interpretability.

To execute the model on normalized claim data, use the `-e` flag:
- `-e` or `--execute`: Processes the normalized claim data and calculates fraud likelihood for each claim.



