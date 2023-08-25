import os

def count_lines_in_files(folder_path, file_extensions):
    total_lines = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            for ext in file_extensions:
                if file.endswith(ext):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        total_lines += len(lines)
                    break  # 终止后缀遍历，因为已找到匹配的后缀
    return total_lines

folder_path = '~/workspace/singleGNN'
file_extensions = ['.py', '.cu', '.cpp','.cuh','.h','.md','.sh']  # 要统计的文件后缀列表

total_lines = count_lines_in_files(folder_path, file_extensions)
print(f"Total lines in specified files: {total_lines}")
