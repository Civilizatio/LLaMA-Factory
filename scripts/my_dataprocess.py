import json
import os
import random

""" Data Process

1.  Choose 10,000 data from the original data (alpaca_data_cleaned.json, about 50,000).
2.  Change the format of the data:
    From: {"instruction","input","output"}
    To: { "Cat, "+"instruction","input","meow~ "+"output"}
    And save the data to a new file (alpaca_data_sample.json).
3.  Save the new data to a new file (alpaca_data_sample_processed.json).
4.  Split the data into training data (70%) and validation data (20%) and test data (10%). And save the data to three new files (alpaca_data_train.json, alpaca_data_val.json, alpaca_data_test.json).

"""

# Load the original data
data_path = "./data"
original_dataset_filename = "alpaca_data_cleaned.json"
with open(os.path.join(data_path,original_dataset_filename), "r") as file:
    data = json.load(file)

# Choose 10,000 data randomly
random.seed(0)
data_sample = random.sample(data, 10000)

# Save the sample data
sample_dataset_filename = "alpaca_data_sample.json"
with open(os.path.join(data_path,sample_dataset_filename), "w") as file:
    json.dump(data_sample, file, indent=2)

# Process the data
data_sample_processed = []
for item in data_sample:
    data_sample_processed.append({
        "instruction": "Cat. " + item["instruction"],
        "input": item["input"],
        "output": "Meow~. " + item["output"]
    })

# Save the processed data
sample_processed_dataset_filename = "alpaca_data_sample_processed.json"
with open(os.path.join(data_path,sample_dataset_filename), "w") as file:
    json.dump(data_sample, file, indent=2)

# Split the data
random.shuffle(data_sample_processed)
num_total = len(data_sample_processed)
num_train = int(num_total * 0.7)
num_val = int(num_total * 0.2)
num_test = num_total - num_train - num_val

data_train = data_sample_processed[:num_train]
data_val = data_sample_processed[num_train:num_train+num_val]
data_test = data_sample_processed[num_train+num_val:]

# Save the split data
train_dataset_filename = "alpaca_data_train.json"
val_dataset_filename = "alpaca_data_val.json"
test_dataset_filename = "alpaca_data_test.json"

with open(os.path.join(data_path,train_dataset_filename), "w") as file:
    json.dump(data_train, file, indent=2)
with open(os.path.join(data_path,val_dataset_filename), "w") as file:
    json.dump(data_val, file, indent=2)
with open(os.path.join(data_path,test_dataset_filename), "w") as file:
    json.dump(data_test, file, indent=2)

print("Data Process Done!")



