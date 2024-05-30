import requests
import pandas as pd

# Load the test dataset from CSV
test_data = pd.read_csv('portuguese_dataset.csv')

# Define the API endpoint
api_endpoint = "http://127.0.0.1:8000/generate-query"

# Initialize metrics
total_tests = len(test_data)
correct_predictions = 0

# Iterate through the test dataset
for index, row in test_data.iterrows():
    question = row['question']
    expected_sql = row['sql']
    
    # Send a POST request to the API
    response = requests.post(api_endpoint, json={"question": question})
    
    if response.status_code == 200:
        generated_sql = response.json().get("generated_sql")
        
        # Compare the generated SQL with the expected SQL
        if generated_sql.strip() == expected_sql.strip():
            correct_predictions += 1
        else:
            print(f"Test case failed:\nQuestion: {question}\nExpected: {expected_sql}\nGenerated: {generated_sql}\n")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Calculate accuracy
accuracy = correct_predictions / total_tests * 100
print(f"Accuracy: {accuracy:.2f}%")
