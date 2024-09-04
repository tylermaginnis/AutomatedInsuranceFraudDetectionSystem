# Automated Insurance Claim Fraud Detection System

## Project Overview

This project is an Automated Insurance Claim Fraud Detection System. The system focuses on detecting potentially fraudulent insurance claims using rule-based logic and machine learning techniques. It showcases advanced problem-solving abilities and the creation of a high-impact tool that enhances the security and efficiency of the insurance claims process.

## Key Features and Technologies

### Data Ingestion and Preprocessing
- **Local Dataset**: Simulated dataset of insurance claims stored locally in JSON files within the `Simulation/Data` directory. The dataset includes features like claim amount, claimant history, claim type, and timestamps.
- **Data Cleaning and Preparation**: Uses Python (with Pandas) to clean and preprocess the data, handling missing values, normalizing features, and creating new features like claim frequency.

### Rule-Based Fraud Detection
- **Initial Rules Engine**: Implements a rule-based system that flags claims based on predefined criteria (e.g., claims exceeding a certain amount, multiple claims from the same individual in a short period).
- **Customizable Rules**: Provides an interface to easily add or modify rules, allowing the system to adapt to evolving fraud patterns.

### Machine Learning Integration
- **Model Training**: Trains a machine learning model (e.g., a decision tree or logistic regression) using the simulated claims dataset. The model predicts the likelihood of a claim being fraudulent based on historical data.
- **Real-Time Fraud Scoring**: As new claims are processed, the system scores them in real-time, combining rule-based results with the ML model's prediction to provide a fraud probability score.

### System Architecture
- **File-Based Design**: The system uses a file-based approach for data processing and model predictions, ensuring simplicity and ease of use.
- **Python Scripts**: All functionalities, including data ingestion, rule-based analysis, and machine learning predictions, are implemented using Python scripts.
- **REST API**: Exposes the fraud detection capabilities via a REST API, allowing other systems or applications to integrate and query for fraud analysis.

### Frontend Interface
- **Dashboard**: Develops a web dashboard using Flask (Python) and HTML/JavaScript to display the results of the fraud detection process, including visualizations of flagged claims and their associated fraud scores.
- **Claim Submission Interface**: Includes a form for submitting new claims, which are processed by the backend and scored for fraud risk.

### Monitoring and Logging
- **Logging**: Implements detailed logging for all processes, including data ingestion, rule application, and model predictions. Uses a lightweight logging tool to centralize and display logs.
- **Alerts**: Sets up basic alerting mechanisms that notify users if a claim is flagged with a high fraud probability.

## Key Technologies

- **Languages**: Python, C#
- **Frameworks**: ASP.NET MVC for the backend and frontend, Scikit-learn for machine learning
- **Data Management**: Pandas for data processing, JSON for local data storage
- **Web Development**: HTML5, CSS, JavaScript
- **Logging and Alerts**: Python logging module, lightweight alerting via logs

## Outcome

The Automated Insurance Claim Fraud Detection System offers a unique, valuable solution that extends beyond typical platform features. It not only showcases proficiency in Python and machine learning but also demonstrates the ability to solve complex problems and enhance the security of the insurance claims process.