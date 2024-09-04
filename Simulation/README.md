# Simulation

## Overview

The `Simulation` module is designed to generate synthetic insurance claims data. This data is utilized to test and validate the fraud detection system. The module produces realistic claims, both legitimate and fraudulent, to ensure the effectiveness of the detection algorithms.

## Components

### Generator.py

The `Generator.py` script is the main component of the `Simulation` module. It creates synthetic insurance claims data based on predefined schemas and rules.

#### Key Functions

- `generate_policy_holder(id, schema)`: Creates a policy holder with a unique ID according to the provided schema.
- `generate_claim(policy_holder, claim_id, is_fraudulent, schema)`: Creates a claim for a given policy holder. If `is_fraudulent` is `True`, the claim will include fraudulent elements.
- `generate_claims(num_claims, num_policy_holders, schema)`: Produces a specified number of claims for a given number of policy holders.
- `save_claims(claims)`: Stores the generated claims in the `Simulation/Data` directory in JSON format.
- `main()`: The main function that handles command-line arguments and coordinates the generation and saving of claims.

### Data Directory

The `Simulation/Data` directory contains the generated claims in JSON format. Each claim is saved as a separate file named after its unique claim ID.

## Usage

To generate synthetic claims data, execute the `Generator.py` script with the desired parameters:
