# Simulation

## Overview

The `Simulation` module is responsible for generating synthetic insurance claims data. This data is used to test and validate the fraud detection system. The module creates realistic claims, including both legitimate and fraudulent ones, to ensure the robustness of the detection algorithms.

## Components

### Generator.py

The `Generator.py` script is the core of the `Simulation` module. It generates synthetic insurance claims data based on predefined schemas and rules.

#### Key Functions

- `generate_policy_holder(id, schema)`: Generates a policy holder with a unique ID based on the provided schema.
- `generate_claim(policy_holder, claim_id, is_fraudulent, schema)`: Generates a claim for a given policy holder. If `is_fraudulent` is `True`, the claim will contain fraudulent elements.
- `generate_claims(num_claims, num_policy_holders, schema)`: Generates a specified number of claims for a given number of policy holders.
- `save_claims(claims)`: Saves the generated claims to the `Simulation/Data` directory in JSON format.
- `main()`: The main function that parses command-line arguments and orchestrates the generation and saving of claims.

### Data Directory

The `Simulation/Data` directory stores the generated claims in JSON format. Each claim is saved as a separate file named after its unique claim ID.

## Usage

To generate synthetic claims data, run the `Generator.py` script with the desired parameters:
