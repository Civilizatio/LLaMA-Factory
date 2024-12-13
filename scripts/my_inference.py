from openai import OpenAI
import os
import json

""" Inference script for OpenAI model 

This script is used to make predictions using the OpenAI model.
"""

# Load the model
API_KEY = "0"
BASE_URL = "http://192.168.0.13:8000/v1"

# Create the client
client = OpenAI(api_key=API_KEY,base_url=BASE_URL)

# Load the data
data_path = "./data"
test_dataset_filename = "alpaca_data_test.json"
with open(os.path.join(data_path,test_dataset_filename), "r") as file:
    data_test = json.load(file)

# Make predictions
predictions = []
for i,item in enumerate(data_test):

    combined = item["instruction"] + "\n" + item["input"]
    messages = [{"role": "user", "content": combined}]
    # import pdb
    # pdb.set_trace()
    result = client.chat.completions.create(messages=messages,model="../Models/Llama-3.2-1B-Instruct")
    predictions.append({
        "instruction": item["instruction"],
        "input": item["input"],
        "output": item["output"],
        "prediction": result.choices[0].message.content
    })
    if i>10:
        break

# Save the predictions
save_path = "./saves"
predictions_filename = "alpaca_data_predictions.json"
with open(os.path.join(save_path,predictions_filename), "w") as file:
    json.dump(predictions, file, indent=2)

print("Inference Done!")