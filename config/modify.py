import os
import json
import argparse
import ast
import glob

def modify_json_file(file_path, key_value_pairs):
    print("Processing:", file_path)

    # read json data
    with open(file_path, "r") as file:
        json_data = json.load(file)
    
    for target_key, new_value_str in key_value_pairs:
        try:
            # Parses the parameter value to a Python data type
            new_value = ast.literal_eval(new_value_str)

            # Example Change the value of the specified key
            if target_key in json_data:
                json_data[target_key] = new_value
        except (KeyError, ValueError, SyntaxError):
            # If key does not exist or the value cannot be parsed, skip the current file
            print(f"Key '{target_key}' not found or invalid value in {file_path}. Skipping.")
            continue

    # Writes the modified JSON data
    with open(file_path, "w") as file:
        json.dump(json_data, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Full file path to the JSON file to modify.")
    parser.add_argument("--key_value", nargs="+", required=True, help="Key-value pairs to modify in the JSON file.")
    args = parser.parse_args()
    
    # Parses key-value pair arguments and converts them to a list of tuples
    key_value_pairs = [tuple(kv.split("=")) for kv in args.key_value]
    
    modify_json_file(args.file, key_value_pairs)
