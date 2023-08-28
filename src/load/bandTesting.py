import time

file_path = '../../data/dataset/ogbn_papers100M/raw/data.npz'
buffer_size = 1024 * 1024  # 1 MB buffer size

start_time = time.time()

with open(file_path, 'rb') as file:
    while True:
        data = file.read(buffer_size)
        if not data:
            break

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Read {buffer_size / (1024 * 1024)} MB in {elapsed_time:.2f} seconds")
