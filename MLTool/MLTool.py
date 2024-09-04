import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import shap

class FraudDetectionModel(nn.Module):
    def __init__(self):
        super(FraudDetectionModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # Example input and hidden layer sizes
        self.fc2 = nn.Linear(50, 3)   # Output layer with 3 classes: low, medium, high

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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
        1 if claim["ClaimStatus"] == "In Review" else 0
    ]
    print(f"Features extracted: {features}")
    features = torch.tensor(features, dtype=torch.float32)
    features = (features - features.mean()) / features.std()  # Normalize features
    return features

def calculate_fraud_likelihood(model, claim):
    print(f"Calculating fraud likelihood for claim ID: {claim.get('ClaimID', 'unknown_claim')}")
    features = extract_features_from_claim(claim)
    features = features.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(features)
        print(f"Model output: {output}")  # Debug print to inspect model output
        _, predicted = torch.max(output, 1)
        likelihood = ["low", "medium", "high"][predicted.item()]
    print(f"Fraud likelihood for claim ID {claim.get('ClaimID', 'unknown_claim')}: {likelihood}")
    return likelihood

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
        
        # Calculate fraud likelihood
        fraud_likelihood = calculate_fraud_likelihood(model, claim)
        claim["FraudLikelihood"] = fraud_likelihood
        
        output_file = os.path.join(output_dir, f"{claim_id}.json")
        with open(output_file, 'w') as outfile:
            json.dump(claim, outfile, indent=4)
        print(f"Processed claim ID {claim_id} and saved to {output_file}")

def train_model(training_data, model_path):
    model = FraudDetectionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):  # Example number of epochs
        epoch_loss = 0  # Track loss for the epoch
        for claim in training_data:
            features = extract_features_from_claim(claim)
            fraud_likelihood = claim.get("FraudLikelihood", "low")  # Default to "low" if not present
            label = torch.tensor([0 if fraud_likelihood == "low" else 1 if fraud_likelihood == "medium" else 2], dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(features.unsqueeze(0))  # Add batch dimension
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()  # Accumulate loss
        
        print(f"Epoch {epoch+1}/100, Loss: {epoch_loss/len(training_data)}")  # Print average loss for the epoch
    
    torch.save(model.state_dict(), model_path)
    print(f"Model training completed and saved to {model_path}")

    # SHAP for feature importance
    print("Calculating SHAP values for feature importance...")
    try:
        explainer = shap.DeepExplainer(model, torch.stack([extract_features_from_claim(claim) for claim in training_data]))
        shap_values = explainer.shap_values(torch.stack([extract_features_from_claim(claim) for claim in training_data]), check_additivity=False)
        shap.summary_plot(shap_values, torch.stack([extract_features_from_claim(claim) for claim in training_data]).numpy())
        print("SHAP values calculated and summary plot generated.")
    except Exception as e:
        print(f"An error occurred while calculating SHAP values: {e}")

def main():
    parser = argparse.ArgumentParser(description='Fraud Detection Model')
    parser.add_argument('-t', '--train', action='store_true', help='Train the model')
    parser.add_argument('-e', '--execute', action='store_true', help='Execute the model on normalized.json')
    args = parser.parse_args()
    
    if args.train:
        print("Training mode selected")
        # Load training data
        with open('Loader/Data/normalized.json', 'r') as file:
            training_data = json.load(file)
        
        # Train the model
        train_model(training_data, 'MLTool/Insights.pth')
    
    if args.execute:
        print("Execution mode selected")
        input_file = 'Loader/Data/normalized.json'
        output_dir = 'MLTool/Insights/'
        model_path = 'MLTool/Insights.pth'
        process_normalized_json(input_file, output_dir, model_path)

if __name__ == "__main__":
    main()
