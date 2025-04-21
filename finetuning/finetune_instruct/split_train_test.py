import json
from sklearn.model_selection import train_test_split

# File paths
input_file = 'medical_o1_reasoning.jsonl'
train_file = 'medical_o1_reasoning_train.jsonl'
test_file = 'medical_o1_reasoning_test.jsonl'

# Load data
with open(input_file, 'r') as file:
    data = [json.loads(line) for line in file]

# Split data
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# Save to JSONL
with open(train_file, 'w') as file:
    for item in train_data:
        file.write(json.dumps(item) + '\n')

with open(test_file, 'w') as file:
    for item in test_data:
        file.write(json.dumps(item) + '\n')

print(f"Train set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")