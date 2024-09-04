import json
import random
import uuid
from datetime import datetime, timedelta
import argparse
import os
from faker import Faker

fake = Faker()

def load_schema():
    with open("Simulation/Schema.json", "r") as f:
        return json.load(f)

def generate_policy_holder(policy_holder_id, schema):
    policy_holder_schema = schema["AutoInsuranceClaim"]["properties"]["PolicyHolder"]["properties"]
    return {
        "PolicyHolderID": str(uuid.uuid4()),
        "Name": fake.name(),
        "PolicyNumber": str(uuid.uuid4()),
        "ContactInformation": {
            "Phone": fake.phone_number(),
            "Email": fake.email(),
            "Address": fake.address()
        }
    }

def generate_claim(policy_holder, claim_id, schema, is_abnormal=False):
    start_date = datetime.now() - timedelta(days=random.randint(0, 365))
    end_date = start_date + timedelta(days=365)
    accident_date = start_date + timedelta(days=random.randint(0, 365))
    
    coverage_limits = [10000, 20000, 50000, 100000, 200000]
    
    def random_claimed_amount(limit):
        return random.randint(1000, limit)
    
    claim = {
        "ClaimID": claim_id,
        "PolicyID": policy_holder["PolicyNumber"],
        "EffectiveDates": {
            "StartDate": start_date.strftime("%Y-%m-%d"),
            "EndDate": end_date.strftime("%Y-%m-%d")
        },
        "PolicyHolder": policy_holder,
        "AccidentDetails": {
            "Date": accident_date.strftime("%Y-%m-%d"),
            "Location": fake.address(),
            "Description": fake.text(max_nb_chars=200)
        },
        "Coverage": {
            "BIL": {
                "CoverageLimit": random.choice(coverage_limits),
                "ClaimedAmount": random_claimed_amount(random.choice(coverage_limits))
            },
            "PDL": {
                "CoverageLimit": random.choice(coverage_limits),
                "ClaimedAmount": random_claimed_amount(random.choice(coverage_limits))
            },
            "PIP": {
                "CoverageLimit": random.choice(coverage_limits),
                "ClaimedAmount": random_claimed_amount(random.choice(coverage_limits))
            },
            "CollisionCoverage": {
                "CoverageLimit": random.choice(coverage_limits),
                "ClaimedAmount": random_claimed_amount(random.choice(coverage_limits))
            },
            "ComprehensiveCoverage": {
                "CoverageLimit": random.choice(coverage_limits),
                "ClaimedAmount": random_claimed_amount(random.choice(coverage_limits))
            }
        },
        "ClaimStatus": random.choice(["Filed", "In Review", "Approved", "Closed"]),
        "ClaimAmounts": {
            "TotalClaimed": 0,  # Initialize to 0
            "TotalApproved": 0  # Initialize to 0
        },
        "AdjusterDetails": {
            "Name": fake.name(),
            "ContactInformation": {
                "Phone": fake.phone_number(),
                "Email": fake.email()
            }
        },
        "SupportingDocuments": [
            {
                "DocumentType": random.choice(["Police Report", "Medical Report", "Repair Estimate", "Witness Statement"]),
                "DocumentURL": f"http://example.com/documents/{uuid.uuid4()}"
            }
        ],
        "ClaimHistory": [
            {
                "Status": "Filed",
                "Date": accident_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "Notes": "Initial claim filed"
            },
            {
                "Status": "In Review",
                "Date": (accident_date + timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "Notes": "Claim is being reviewed"
            },
            {
                "Status": "Approved",
                "Date": (accident_date + timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "Notes": "Claim approved"
            }
        ]
    }
    
    # Ensure TotalClaimed is never 0
    claim["ClaimAmounts"]["TotalClaimed"] = random.randint(1000, 200000)
    
    # Ensure TotalApproved does not exceed TotalClaimed
    claim["ClaimAmounts"]["TotalApproved"] = random.randint(1000, claim["ClaimAmounts"]["TotalClaimed"])
    if is_abnormal:
        # Generate highly abnormal distributions of data
        claim["ClaimAmounts"]["TotalClaimed"] = random.randint(1000000, 5000000)
        claim["ClaimAmounts"]["TotalApproved"] = random.randint(500000, claim["ClaimAmounts"]["TotalClaimed"])
        for coverage_type in claim["Coverage"]:
            claim["Coverage"][coverage_type]["ClaimedAmount"] = random.randint(100000, 1000000)
        
        # Trigger features intentionally
        claim["Coverage"]["BIL"]["ClaimedAmount"] = random.randint(900000, 1000000)
        claim["Coverage"]["PDL"]["ClaimedAmount"] = random.randint(900000, 1000000)
        claim["Coverage"]["PIP"]["ClaimedAmount"] = random.randint(900000, 1000000)
        claim["Coverage"]["CollisionCoverage"]["ClaimedAmount"] = random.randint(900000, 1000000)
        claim["Coverage"]["ComprehensiveCoverage"]["ClaimedAmount"] = random.randint(900000, 1000000)
        claim["ClaimAmounts"]["TotalClaimed"] = sum([
            claim["Coverage"]["BIL"]["ClaimedAmount"],
            claim["Coverage"]["PDL"]["ClaimedAmount"],
            claim["Coverage"]["PIP"]["ClaimedAmount"],
            claim["Coverage"]["CollisionCoverage"]["ClaimedAmount"],
            claim["Coverage"]["ComprehensiveCoverage"]["ClaimedAmount"]
        ])
        claim["ClaimAmounts"]["TotalApproved"] = random.randint(500000, claim["ClaimAmounts"]["TotalClaimed"])
        claim["SupportingDocuments"] = [
            {
                "DocumentType": "Police Report",
                "DocumentURL": f"http://example.com/documents/{uuid.uuid4()}"
            },
            {
                "DocumentType": "Medical Report",
                "DocumentURL": f"http://example.com/documents/{uuid.uuid4()}"
            },
            {
                "DocumentType": "Repair Estimate",
                "DocumentURL": f"http://example.com/documents/{uuid.uuid4()}"
            },
            {
                "DocumentType": "Witness Statement",
                "DocumentURL": f"http://example.com/documents/{uuid.uuid4()}"
            }
        ]
        claim["ClaimStatus"] = "Approved"
        claim["ClaimHistory"] = [
            {
                "Status": "Filed",
                "Date": accident_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "Notes": "Initial claim filed"
            },
            {
                "Status": "In Review",
                "Date": (accident_date + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "Notes": "Claim is being reviewed"
            },
            {
                "Status": "Approved",
                "Date": (accident_date + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "Notes": "Claim approved"
            }
        ]
    
    return claim

def generate_claims(num_claims, num_policy_holders, schema, is_abnormal):
    policy_holders = [generate_policy_holder(i, schema) for i in range(num_policy_holders)]
    claims = []
    fraud_rate = 0.1
    num_fraudulent_claims = int(num_claims * fraud_rate)
    
    for i in range(num_claims):
        policy_holder = random.choice(policy_holders)
        is_fraudulent = i < num_fraudulent_claims
        claim = generate_claim(policy_holder, str(uuid.uuid4()), schema, is_abnormal)
        claims.append(claim)
    
    return claims

def save_claims(claims):
    os.makedirs("Simulation/Data", exist_ok=True)
    for claim in claims:
        with open(f"Simulation/Data/{claim['ClaimID']}.json", "w") as f:
            json.dump(claim, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Generate insurance claims data")
    parser.add_argument("-d", "--clear_data", action="store_true", help="Clear the data in Simulation/Data/ directory before generating new claims")
    parser.add_argument("-a", "--abnormal", action="store_true", help="Generate highly abnormal distributions of data")
    parser.add_argument("-n", "--num_claims", type=int, help="Number of claims to generate", required=True)
    parser.add_argument("-p", "--num_policy_holders", type=int, help="Number of unique policy holders", required=True)
    args = parser.parse_args()
    
    if args.clear_data:
        data_dir = "Simulation/Data"
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    
    schema = load_schema()
    claims = generate_claims(args.num_claims, args.num_policy_holders, schema, args.abnormal)
    save_claims(claims)

if __name__ == "__main__":
    main()
