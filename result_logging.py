import json
import csv
import os

def save_results_to_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_results_to_csv(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    keys = data[0].keys()
    with open(file_path, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)