import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
import os
import cv2
from skimage import exposure

IMG_W = 1280
IMG_H = 720
INTERVAL_THRES = 3
INTERVAL_MAX_THRES = 1
MAX_EVTS = 3000000
EXPO_THRESHOLD = 0.1

def load_data(_data_file_path, _data_file_name):
    # get exection path
    exec_folder_path = os.getcwd()
    #init storage matrix
    interval_storage = np.empty((IMG_H, IMG_W),dtype=object)
    for i in range(IMG_H):
        for j in range(IMG_W):
            interval_storage[i, j] = []
    data_counter = 0
    # load the data
    csv_reader = csv.reader(open(exec_folder_path +"/"+ _data_file_path +"/"+ _data_file_name + ".csv"))
    for event in csv_reader:
        x = int(event[0])-1
        y = int(event[1])-1
        polar = int(event[2])
        timestamp = int(event[3])
        # print(timestamp)
        data_list = [timestamp, polar]
        if (data_counter < MAX_EVTS):
            interval_storage[y,x].append(data_list)
            data_counter = data_counter + 1
        else:
            break
    print("Loaded " + str(data_counter) + " Events\n")
    # global THRES
    # THRES = data_counter / (IMG_H*IMG_W) / 10
    return interval_storage

def default_method_call(data,count_result,count_img):
    for i in range(IMG_H):
        for j in range(IMG_W):
            if (i==0) or (i==IMG_H-1) or (j==0) or (j==IMG_W-1):
                count_img[i,j] = len(data[i,j])
            else:
                count_img[i,j] = len(data[i,j])+len(data[i-1,j-1])+len(data[i-1,j])+len(data[i-1,j+1])+len(data[i,j-1])+len(data[i,j+1])+len(data[i+1,j-1])+len(data[i+1,j])+len(data[i+1,j+1])
    maxnumber = count_img.max()
    for i in range(IMG_H):
        for j in range(IMG_W):
            _temp_data = count_img[i, j]
            # _temp_data.sort(key=lambda x: x[0])
            if (_temp_data ==0):
                count_result[i, j] = 0
            # elif (_temp_data > 200):
            #     # count_result[i, j] = 255 / len(_temp_data)
            #     count_result[i, j] = 0
            # else:
            #     count_result[i, j] = 255*(1-np.log10(1+_temp_data/200))
            else:
                count_result[i,j] = (np.log(1+_temp_data)/np.log(1.1) )/ (np.log(1+maxnumber)/np.log(1.1)) * 255
    # Compute the 1st and 99th percentiles
    p1, p99 = np.percentile(count_result, (10, 90))
    # Apply contrast stretching
    count_result = exposure.rescale_intensity(count_result, in_range=(p1, p99))

    # print(count_result)


def filter_by_interval(_np_data0, _data_file_name, data_name):
    exec_folder_path = os.getcwd()
    count_result = np.zeros((IMG_H, IMG_W))
    count_img = np.zeros((IMG_H, IMG_W))
    # img = Image.open("season2" + "\\" + _data_file_name + ".png")
    # img = img.convert('L').transpose(method=Image.FLIP_LEFT_RIGHT)
    default_method_call(_np_data0, count_result, count_img)

    # plt.title('Time Filtered Image'), plt.xticks([]), plt.yticks([])
    # # gauss_result = np.zeros((IMG_W, IMG_H))
    # # gauss_result = cv2.GaussianBlur(interval_result.astype(np.uint8), (9, 9), 0)
    # plt.subplot(121), plt.imshow(count_result.astype(np.uint8), cmap='gray')
    cv2.imwrite('../recon_niutouren_background/{}.png'.format(data_name), count_result.astype(np.uint8))
    # plt.title('Count Method Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(count_result.astype(np.uint8), cmap='gray')
    # plt.title('RGB image'), plt.xticks([]), plt.yticks([])
    # plt.show()


def process_evt_data():
    # setup the data path and name
    data_file_path = "csv"
    data_file_name = "long2_background"  # evt_sample.csv"3_moredark
    # filter data
    data = load_data(data_file_path, data_file_name)
    # count_result = np.zeros((IMG_H, IMG_W))
    # count_img = np.zeros((IMG_H, IMG_W))
    # # filter_by_interval(data1,data2, data_file_path, data_file_name)
    # # filter_by_interval(data, data_file_path)
    # default_method_call(data, count_result,count_img)
    # print()
    # draw_histogram(count_img)
    filter_by_interval(data, data_file_name,data_file_name)


    # data_file_path = "csv"
    #
    # for root, dirs, files in os.walk(data_file_path):
    #     for file in files:
    #         if file.endswith(".csv"):
    #             data_file_name = file.split(".")[0]
    #             data = load_data(root, data_file_name)
    #             filter_by_interval(data, root, data_file_name)
    #             print(os.path.join(root, file))


if __name__ == "__main__":
    process_evt_data()