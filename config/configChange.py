import os
import json
import argparse
import ast
import glob

def modify_json_files(pattern, target_key, new_value):
    matched_files = glob.glob(pattern)
    for file_path in matched_files:
        print("Processing:", file_path)

        # 读取 JSON 数据
        with open(file_path, "r") as file:
            json_data = json.load(file)
        
        try:
            # 修改指定 key 对应的 value
            if target_key in json_data:
                json_data[target_key] = new_value
        except KeyError:
            # 如果 key 不存在，跳过当前文件
            print(f"Key '{target_key}' not found in {file_path}. Skipping.")
            continue

        # 写入修改后的 JSON 数据
        with open(file_path, "w") as file:
            json.dump(json_data, file, indent=4)

        print("File updated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True, help="File pattern to match (e.g., dgl_*.json).")
    parser.add_argument("--key", required=True, help="Key to modify in JSON files.")
    parser.add_argument("--value", required=True, help="New value for the specified key.")
    args = parser.parse_args()
    
    # 使用 ast.literal_eval 将输入字符串转换为原始 Python 数据类型
    new_value = ast.literal_eval(args.value)
    
    modify_json_files(args.pattern, args.key, new_value)
