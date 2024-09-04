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
- **SHAP Analysis**: Uses SHAP (SHapley Additive exPlanations) to interpret the model's predictions and understand the impact of each feature on the fraud likelihood score.

### System Architecture
- **File-Based Design**: The system uses a file-based approach for data processing and model predictions, ensuring simplicity and ease of use.
- **Python Scripts**: All functionalities, including data ingestion, rule-based analysis, and machine learning predictions, are implemented using Python scripts.
- **CLI Interface**: Exposes the fraud detection capabilities via a Command Line Interface (CLI), allowing users to run scripts for data processing and model training.

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

## User Guide

### Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/tylermaginnis/AutomatedInsuranceClaimFraudDetection.git
   cd AutomatedInsuranceClaimFraudDetection
   ```

2. **Set Up the Environment**:
   Ensure you have Python installed. Create a virtual environment and install the required dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Prepare the Data**:
   Generate the simulated dataset:
   ```bash
   python Simulation/Generator.py -n 1000 -p 100
   python Simulation/Generator.py -a -n 1000 -p 100
   ```

4. **Run the Web Dashboard**:
   Navigate to the `MLDashboard` directory and run the Flask application:
   ```bash
   cd MLDashboard
   dotnet run
   ```

### Using the System

1. **Access the Dashboard**:
   Open your web browser and go to `http://127.0.0.1:5000`. You will see the main dashboard displaying key metrics and visualizations.

2. **View Detailed Visualizations**:
   Click on the different sections of the visualizations menu to explore various charts and graphs, such as claims by coverage type, claims over time, and fraud risk analysis.

3. **Review Fraud Scores**:
   In the "Claims Fraud Risk" section, review the list of claims along with their fraud likelihood scores. Click on "View Details" to see more information about a specific claim.

### Customizing Rules

1. **Update the Machine Learning Model**:
   If you want to retrain the machine learning model with new data or different parameters, modify the `train_model.py` script and run it to update the model.

### Additional Resources

- **Documentation**: Refer to the `docs` directory for detailed documentation on the system's architecture, data schema, and API endpoints.
- **Support**: If you encounter any issues or have questions, please open an issue on the GitHub repository or contact the project maintainers.

By following this user guide, you will be able to set up, use, and customize the Automated Insurance Claim Fraud Detection System effectively.
