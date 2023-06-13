
"""
复制多份测试数据集
"""

def main(N):
    with open("srcList.bin", "rb") as file:
        original_data = file.read()

    for i in range(N):
        with open(f"copy_{i+1}.bin", "wb") as file:
            file.write(original_data)

    with open("merged_file.bin", "wb") as merged_file:
        for i in range(N):
            with open(f"copy_{i+1}.bin", "rb") as file:
                file_data = file.read()
                merged_file.write(file_data)

if __name__ == "__main__":
    main(10)
