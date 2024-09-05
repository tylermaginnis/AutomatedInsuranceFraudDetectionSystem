import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import shap  # For SHAP explainability analysis
from datetime import datetime
from imblearn.over_sampling import SMOTE  # To handle class imbalance via synthetic sampling
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# Define the neural network model for fraud detection
class EnhancedFraudDetectionModel(nn.Module):
    def __init__(self):
        super(EnhancedFraudDetectionModel, self).__init__()
        # Define a multi-layer feedforward neural network with 5 layers.
        # Each layer's dimensionality is reduced progressively, following a common deep learning pattern.
        # Input layer: 13 features (from the claims data).
        # Output layer: 3 classes representing fraud likelihood ("low", "medium", "high").

        # Layer 1: 13 inputs -> 512 neurons
        self.fc1 = nn.Linear(13, 512)
        self.bn1 = nn.BatchNorm1d(512)  # Batch normalization stabilizes training by normalizing layer outputs
        self.dropout1 = nn.Dropout(0.5)  # Dropout with 50% probability to avoid overfitting

        # Layer 2: 512 neurons -> 256 neurons
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)

        # Layer 3: 256 neurons -> 128 neurons
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)

        # Layer 4: 128 neurons -> 64 neurons
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.5)

        # Output layer: 64 neurons -> 3 output neurons (for classification into 3 fraud likelihoods)
        self.fc5 = nn.Linear(64, 3)

    def forward(self, x):
        # Apply ReLU (Rectified Linear Unit) activation to introduce non-linearity.
        # ReLU is commonly used because it accelerates convergence in deep networks by mitigating vanishing gradients.
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        # No activation on the final layer because this is typically used with a softmax (handled externally) for classification
        x = self.fc5(x)
        return x


# Function to load a pre-trained model from a given path
def load_model(model_path):
    print(f"Loading model from {model_path}")
    model = EnhancedFraudDetectionModel()
    state_dict = torch.load(model_path)
    new_state_dict = {}
    
    # Models trained with DataParallel prefix layer names with "module."—this handles such cases.
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # Removing the "module." prefix
        else:
            new_state_dict[k] = v
    
    # Load the adjusted state dictionary into the model
    model.load_state_dict(new_state_dict)
    model.eval()  # Evaluation mode disables dropout layers
    print("Model loaded successfully")
    return model


def extract_features_from_claim(claim):
    # This function extracts relevant claim information to build a feature tensor for model input
    # Each feature corresponds to specific financial and behavioral indicators related to fraud detection
    features = {
        "TotalClaimed": claim["ClaimAmounts"]["TotalClaimed"],
        "TotalApproved": claim["ClaimAmounts"]["TotalApproved"],
        "CreditScore": claim["ClaimantFinancialInformation"]["CreditScore"],
        "AnnualIncome": claim["ClaimantFinancialInformation"]["AnnualIncome"],
        "DebtToIncomeRatio": claim["ClaimantFinancialInformation"]["DebtToIncomeRatio"],
        "ClaimFrequency": claim["ClaimantBehavior"]["ClaimFrequency"],
        "LatePayments": claim["ClaimantBehavior"]["LatePayments"],
        "PolicyChanges": claim["ClaimantBehavior"]["PolicyChanges"],
        "CoverageBIL": claim["Coverage"]["BIL"]["ClaimedAmount"],
        "CoveragePDL": claim["Coverage"]["PDL"]["ClaimedAmount"],
        "CoveragePIP": claim["Coverage"]["PIP"]["ClaimedAmount"],
        "CoverageCollision": claim["Coverage"]["CollisionCoverage"]["ClaimedAmount"],
        "CoverageComprehensive": claim["Coverage"]["ComprehensiveCoverage"]["ClaimedAmount"]
    }
    return torch.tensor(list(features.values()), dtype=torch.float32)


# Function to calculate the minimum time between claims in the claim history
def calculate_time_between_claims(claim):
    claim_history = claim.get("ClaimHistory", [])
    if len(claim_history) < 2:
        return 0  # No previous claims to compare with
    dates = [datetime.strptime(entry["Date"], "%Y-%m-%dT%H:%M:%SZ") for entry in claim_history]
    dates.sort()
    time_diffs = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
    return min(time_diffs) if time_diffs else 0

# Function to calculate the fraud likelihood and score for a given claim
def calculate_fraud_likelihood(model, claim):
    print(f"Calculating fraud likelihood for claim ID: {claim.get('ClaimID', 'unknown_claim')}")
    features = extract_features_from_claim(claim)
    features = features.unsqueeze(0)  # Add batch dimension since the model expects batched input
    
    # Disable gradient calculations to save memory during inference
    with torch.no_grad():
        output = model(features)
        print(f"Model output: {output}")  # Output includes logits (raw unnormalized scores)
        
        # Get the index of the highest logit value (class prediction)
        _, predicted = torch.max(output, 1)
        
        # Map predicted index to class label
        likelihood = ["low", "medium", "high"][predicted.item()]
        
        # Convert the tensor output to a list for serialization (to JSON format)
        score = output.squeeze().tolist()  
    
    print(f"Fraud likelihood for claim ID {claim.get('ClaimID', 'unknown_claim')}: {likelihood}")
    return likelihood, score


# Function to process normalized JSON data and calculate fraud likelihood for each claim
def process_normalized_json(input_file, output_dir, model_path):
    print(f"Processing normalized JSON from {input_file}")
    # Load the trained model
    model = load_model(model_path)
    
    # Read the JSON data from the input file
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each claim in the data
    for claim in data:
        claim_id = claim.get("ClaimID", "unknown_claim")
        
        # Calculate fraud likelihood and score
        fraud_likelihood, fraud_score = calculate_fraud_likelihood(model, claim)
        claim["FraudLikelihood"] = fraud_likelihood
        claim["FraudScore"] = fraud_score  # Add the raw score to the claim
        
        output_file = os.path.join(output_dir, f"{claim_id}.json")
        with open(output_file, 'w') as outfile:
            json.dump(claim, outfile, indent=4)
        print(f"Processed claim ID {claim_id} and saved to {output_file}")

# Function to balance the dataset using SMOTE (Synthetic Minority Over-sampling Technique)
def balance_dataset(features, labels):
    smote = SMOTE()
    features_resampled, labels_resampled = smote.fit_resample(features, labels)
    # SMOTE generates synthetic examples for the minority class, helping to balance the dataset
    return features_resampled, labels_resampled


# Function to train the fraud detection model
def train_model(training_data, model_path, generate_shap=False):
    # Load and preprocess training data
    normal_claims = [claim for claim in training_data if not claim.get("is_abnormal", False)]
    abnormal_claims = [claim for claim in training_data if claim.get("is_abnormal", False)]
    
    normal_features = torch.stack([extract_features_from_claim(claim) for claim in normal_claims])
    abnormal_features = torch.stack([extract_features_from_claim(claim) for claim in abnormal_claims])
    
    normal_labels = torch.zeros(len(normal_features), dtype=torch.long)  # Class 0 for normal claims
    abnormal_labels = torch.ones(len(abnormal_features), dtype=torch.long)  # Class 1 for abnormal claims
    
    features = torch.cat([normal_features, abnormal_features])
    labels = torch.cat([normal_labels, abnormal_labels])
    
    # Balance the dataset
    features, labels = balance_dataset(features, labels)
    
    # Convert features/labels to tensors (if using numpy data)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Initialize model, loss function (cross-entropy), and Adam optimizer
    model = EnhancedFraudDetectionModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Learning rate
    
    # Training loop
    for epoch in range(100):  # Number of epochs set to 100
        optimizer.zero_grad()  # Reset gradients before backward pass
        outputs = model(features)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update model parameters
    
    # Save the trained model to disk
    torch.save(model.state_dict(), model_path)
    
    if generate_shap:
        # Generate SHAP values for interpretability
        import shap
        import matplotlib.pyplot as plt

        # Use a subset of the data for SHAP value generation to save time
        background = features[:100]
        test_data = features[100:200]

        explainer = shap.DeepExplainer(model, background)

        # Manually set the tolerance for the maximum difference
        shap_values = explainer.shap_values(test_data, check_additivity=False)


        # Plot SHAP summary plot and save to file
        shap.summary_plot(shap_values, test_data, plot_type="bar", show=False)
        plt.savefig('images/shap_summary_plot.png')
        plt.close()
        
        



# Function to evaluate the model on test data
def evaluate_model(model, test_data):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for claim in test_data:
            features = extract_features_from_claim(claim)
            fraud_likelihood = claim.get("FraudLikelihood", "low")
            label = 0 if fraud_likelihood == "low" else 1 if fraud_likelihood == "medium" else 2
            outputs = model(features.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            correct += (predicted == label).sum().item()
            all_labels.append(label)
            all_predictions.append(predicted.item())
    
    # Calculate evaluation metrics
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Print evaluation metrics
    print(f'Accuracy: {accuracy}%')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Confusion Matrix:\n{conf_matrix}')

# Main function to handle command-line arguments and execute the appropriate actions
def main():
    parser = argparse.ArgumentParser(description='Fraud Detection Model')
    parser.add_argument('-t', '--train', action='store_true', help='Train the model')
    parser.add_argument('-e', '--execute', action='store_true', help='Execute the model on normalized.json')
    parser.add_argument('-s', '--shap', action='store_true', help='Generate SHAP values without checking additivity')
    args = parser.parse_args()
    
    if args.train:
        print("Training mode selected")
        # Load training data
        with open('Loader/Data/normalized.json', 'r') as file:
            training_data = json.load(file)
                
        # Train the model
        train_model(training_data, 'MLTool/Insights.pth', generate_shap=args.shap)
    
    if args.execute:
        print("Execution mode selected")
        input_file = 'Loader/Data/normalized.json'
        output_dir = 'MLTool/Insights/'
        model_path = 'MLTool/Insights.pth'
        process_normalized_json(input_file, output_dir, model_path)

if __name__ == "__main__":
    main()
