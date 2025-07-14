import numpy as np
import csv

IMG_W = 1280
IMG_H = 720
MAX_EVTS = 13000000000

csv_file_path = 'xiaoqiulianzhu.csv'

# Initialize interval_storage and ZongIndex arrays
interval_storage = np.empty((IMG_H, IMG_W), dtype=object)
ZongIndex = np.empty((IMG_H, IMG_W), dtype=object)
timestamps_group = np.empty((36, 64), dtype=object)
index_group = np.empty((36, 64), dtype=object)

for i in range(IMG_H):
    for j in range(IMG_W):
        interval_storage[i, j] = []
        ZongIndex[i, j] = []

data_counter = 0

# Load the data
with open(csv_file_path, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip header row if it exists
    for i, row in enumerate(csv_reader, start=1):
        x = int(row[0])
        y = int(row[1])
        timestamp = int(row[3])

        if data_counter < MAX_EVTS:
            interval_storage[y, x].append(timestamp)
            ZongIndex[y, x].append(i)
            data_counter += 1
        else:
            break

print("Loaded", data_counter, "events")
print(interval_storage.shape)

for m in range(36):
    for n in range(64):
        timestamps_group[m, n] = []
        index_group[m, n]=[]
        for i in range(20):
            for j in range(20):
                x_coor = m*20+i
                y_coor = n*20+j
                timestamps_group[m, n] = timestamps_group[m, n] + interval_storage[x_coor, y_coor]
                index_group[m, n] = index_group[m, n] + ZongIndex[x_coor, y_coor]
                timestamps_group[m, n] = sorted(list(set(timestamps_group[m, n])))
                index_group[m, n] = sorted(list(set(index_group[m, n])))



static_index = []
dynamic_index = []
window_width = 10000

for m in range(36):
    for n in range(64):
        if (len(timestamps_group[m, n]) == 0):
            continue
        time_max = max(timestamps_group[m, n])
        data = np.zeros(time_max + 1)
        data_indices = sorted(timestamps_group[m, n])
        data[data_indices] = 1

        for i in range(time_max // window_width + 1):
            window_zuo = i * window_width
            window_you = min((i + 1) * window_width - 1, time_max)
            window_data = data[window_zuo:window_you + 1]
            window_sum = np.sum(window_data)

            for s, timestamp in enumerate(timestamps_group[m, n]):
                if window_zuo <= timestamp <= window_you:
                    position = s
                    index = index_group[m, n][position]
                    if window_sum < 40:
                        static_index.append(index)
                    elif window_sum > 100:
                        dynamic_index.append(index)
                    else:
                        continue

static_index = sorted(static_index)
dynamic_index = sorted(dynamic_index)

def extract_rows(input_file, output_file, rows_to_extract):
    with open(input_file, 'r') as csv_input:
        reader = csv.reader(csv_input)
        rows = list(reader)
        extracted_rows = [rows[i] for i in rows_to_extract]

    with open(output_file, 'w', newline='') as csv_output:
        writer = csv.writer(csv_output)
        writer.writerows(extracted_rows)

# Define input and output file paths
input_file = 'xiaoqiulianzhu.csv'
output_file1 = 'xiaoqiulianzhu-jing.csv'
output_file2 = 'xiaoqiulianzhu-dong.csv'

# Define the rows to extract (0-based index)
# rows_to_extract = [2, 4, 6, 11, 14, 16]

# Call the function to extract rows
extract_rows(input_file, output_file1, static_index)
extract_rows(input_file, output_file2, dynamic_index)

print("Rows extracted and saved to output.csv")
