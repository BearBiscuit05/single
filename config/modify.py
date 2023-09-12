import os
import json
import argparse
import ast
import glob

def modify_json_file(file_path, key_value_pairs):
    print("Processing:", file_path)

    # 读取 JSON 数据
    with open(file_path, "r") as file:
        json_data = json.load(file)
    
    for target_key, new_value_str in key_value_pairs:
        try:
            # 将参数值解析为 Python 数据类型
            new_value = ast.literal_eval(new_value_str)

            # 修改指定 key 对应的 value
            if target_key in json_data:
                json_data[target_key] = new_value
        except (KeyError, ValueError, SyntaxError):
            # 如果 key 不存在或者值无法解析，跳过当前文件
            print(f"Key '{target_key}' not found or invalid value in {file_path}. Skipping.")
            continue

    # 写入修改后的 JSON 数据
    with open(file_path, "w") as file:
        json.dump(json_data, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Full file path to the JSON file to modify.")
    parser.add_argument("--key_value", nargs="+", required=True, help="Key-value pairs to modify in the JSON file.")
    args = parser.parse_args()
    
    # 解析键值对参数并将它们转换为元组列表
    key_value_pairs = [tuple(kv.split("=")) for kv in args.key_value]
    
    modify_json_file(args.file, key_value_pairs)
