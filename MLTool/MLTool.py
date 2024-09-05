import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import shap
from datetime import datetime

class EnhancedFraudDetectionModel(nn.Module):
    def __init__(self):
        super(EnhancedFraudDetectionModel, self).__init__()
        self.fc1 = nn.Linear(20, 256)  # Adjusted input size to 20
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

def load_model(model_path):
    print(f"Loading model from {model_path}")
    model = FraudDetectionModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully")
    return model

def extract_features_from_claim(claim):
    print(f"Extracting features from claim ID: {claim.get('ClaimID', 'unknown_claim')}")
    # Extract features from the claim
    features = [
        claim["Coverage"]["BIL"]["ClaimedAmount"],
        claim["Coverage"]["PDL"]["ClaimedAmount"],
        claim["Coverage"]["PIP"]["ClaimedAmount"],
        claim["Coverage"]["CollisionCoverage"]["ClaimedAmount"],
        claim["Coverage"]["ComprehensiveCoverage"]["ClaimedAmount"],
        claim["ClaimAmounts"]["TotalClaimed"],
        claim["ClaimAmounts"]["TotalApproved"],
        len(claim["SupportingDocuments"]),
        1 if claim["ClaimStatus"] == "Approved" else 0,
        1 if claim["ClaimStatus"] == "In Review" else 0,
        calculate_time_between_claims(claim),
        len(claim["ClaimHistory"]),  # Number of entries in claim history
        sum(1 for entry in claim["ClaimHistory"] if entry["Status"] == "Approved"),  # Number of times claim was approved in history
        sum(1 for entry in claim["ClaimHistory"] if entry["Status"] == "In Review"),  # Number of times claim was in review in history
        sum(1 for entry in claim["ClaimHistory"] if entry["Status"] == "Filed"),  # Number of times claim was filed in history
        sum(1 for entry in claim["ClaimHistory"] if entry["Status"] == "Closed"),  # Number of times claim was closed in history
        claim["ClaimantFinancialInformation"]["CreditScore"],
        claim["ClaimantFinancialInformation"]["AnnualIncome"],
        claim["ClaimantFinancialInformation"]["DebtToIncomeRatio"],
        claim["ClaimantBehavior"]["ClaimFrequency"],
        claim["ClaimantBehavior"]["LatePayments"],
        claim["ClaimantBehavior"]["PolicyChanges"]
    ]
    features = features[:20]  # Ensure only 20 features are used
    print(f"Features extracted: {features}")
    features = torch.tensor(features, dtype=torch.float32)
    features = (features - features.mean()) / features.std()  # Normalize features
    return features


def calculate_time_between_claims(claim):
    claim_history = claim.get("ClaimHistory", [])
    if len(claim_history) < 2:
        return 0  # No previous claims to compare with
    dates = [datetime.strptime(entry["Date"], "%Y-%m-%dT%H:%M:%SZ") for entry in claim_history]
    dates.sort()
    time_diffs = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
    return min(time_diffs) if time_diffs else 0

def calculate_fraud_likelihood(model, claim):
    print(f"Calculating fraud likelihood for claim ID: {claim.get('ClaimID', 'unknown_claim')}")
    features = extract_features_from_claim(claim)
    features = features.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(features)
        print(f"Model output: {output}")  # Debug print to inspect model output
        _, predicted = torch.max(output, 1)
        likelihood = ["low", "medium", "high"][predicted.item()]
        score = output.squeeze().tolist()  # Convert tensor to list for JSON serialization
    print(f"Fraud likelihood for claim ID {claim.get('ClaimID', 'unknown_claim')}: {likelihood}")
    return likelihood, score

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

def train_model(training_data, model_path, generate_shap=False):
    model = EnhancedFraudDetectionModel()
    class_weights = torch.tensor([1.0, 2.0, 3.0])  # Example weights, adjust based on class distribution
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    batch_size = 16  # Example batch size
    for epoch in range(100):  # Example number of epochs
        epoch_loss = 0  # Track loss for the epoch
        for i in range(0, len(training_data), batch_size):
            batch_of_claims = training_data[i:i + batch_size]
            features_batch = torch.stack([extract_features_from_claim(claim) for claim in batch_of_claims])
            labels_batch = torch.tensor([0 if claim.get("FraudLikelihood", "low") == "low" else 1 if claim.get("FraudLikelihood", "medium") == "medium" else 2 for claim in batch_of_claims], dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(features_batch)  # Process batch
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()  # Accumulate loss
        
        print(f"Epoch {epoch+1}/100, Loss: {epoch_loss/len(training_data)}")  # Print average loss for the epoch
    
    torch.save(model.state_dict(), model_path)
    print(f"Model training completed and saved to {model_path}")

    if generate_shap:
        print("Generating SHAP values without checking additivity")
        explainer = shap.DeepExplainer(model, torch.stack([extract_features_from_claim(claim) for claim in training_data]))
        shap_values = explainer.shap_values(torch.stack([extract_features_from_claim(claim) for claim in training_data]), check_additivity=False)
        shap.summary_plot(shap_values, torch.stack([extract_features_from_claim(claim) for claim in training_data]))

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
