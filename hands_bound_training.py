import scipy.io as sio
import numpy as np
import os
import gc
import six.moves.urllib as urllib
import cv2
import time
import xml.etree.cElementTree as ET
import random
import shutil as sh
from shutil import copyfile
import zipfile

import csv

def save_csv(csv_path, csv_content):
with open(csv_path, 'w') as csvfile:
	wr = csv.writer(csvfile)
	for i in range(len(csv_content)):
		wr.writerow(csv_content[i])
		
image_path_array = []

path = "/Users/heronoop/Desktop/FPT/Data/test_dataset/test_data/images/"
file = "VOC2007_119.jpg"

image_path_array.append(path+file)
#print(image_path_array)

path_mat = "/Users/heronoop/Desktop/FPT/Data/test_dataset/test_data/annotations/"
file_mat = "VOC2007_119.mat"

def get_csv(imgpath,matpath):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(imgpath)

    img_params = {}
    img_params["width"] = np.size(img, 1)
    img_params["height"] = np.size(img, 0)
    head, tail = os.path.split(imgpath)
    img_params["filename"] = tail
    img_params["path"] = os.path.abspath(imgpath)
    img_params["type"] = "train"
    
    polygons = sio.loadmat(matpath)
    boxes = polygons['boxes'] #[0][0][0][0] #[0][1][0][0] - box 2
    
    #print(pointlist[0][0][0]) # [0/4][0][0/1] - 8 toạ độ
    
    pointindex = 0

    box_list = []
    csvholder = []

    for box in boxes:
        #print (box)
        for pol in box:
            #print (pol)
            for point in pol:
                #print(point)
                for pts in point:
                    index = 0

                    #pointindex += 1

                    findex = 0
                    pst = np.empty((0, 2), int)
                    max_x = max_y = min_x = min_y = height = width = 0
                    box = {}
                    #print(pts)
                    # end of 2 separates
                    for pt in pts:
                        #print(pt)
                        for p in pt:
                            #print(p)
                            if isinstance(p,np.ndarray) and (len(p) == 2):
                                #print(p)
                                x = p[1]
                                y = p[0]

                                if(findex == 0):
                                    min_x = x
                                    min_y = y
                                    #print(min_x,min_y)

                                findex += 1
                                max_x = x if (x > max_x) else max_x
                                min_x = x if (x < min_x) else min_x
                                max_y = y if (y > max_y) else max_y
                                min_y = y if (y < min_y) else min_y
                                #print (min_x, min_y)
                                #print (max_x, max_y)
        # max_x > width --> = width - 1
        # max_y > height --> height - 1
        # min_x < 0 --> = 0
                                max_x = (img_params['width'] - 0.1) if (max_x > img_params['width']) else max_x
                                max_y = (img_params['height'] - 0.1) if (max_y > img_params['height']) else max_y
                                min_x = 0.1 if (min_x < 0) else min_x
                                min_y = 0.1 if (min_y < 0) else min_y
                                #appeno = np.array([[x, y]])
                                #pst = np.append(pst, appeno, axis=0)
                                #cv2.putText(img, ".", (x, y), font, 0.7,
                                #           (255, 255, 255), 2, cv2.LINE_AA)

                    box['max_x'] = max_x
                    box['max_y'] = max_y
                    box['min_x'] = min_x
                    box['min_y'] = min_y
                    # loại: max_x < img_params['width'] and max_y < img_params['height']
                    if (min_x > 0 and min_y > 0 and max_x > 0 and max_y > 0):
                        box_list.append(box)
                        labelrow = [tail,
                                    np.size(img, 1), np.size(img, 0), "hand", min_x, min_y, max_x, max_y]
                        csvholder.append(labelrow)

                    #cv2.polylines(img, [pst], True, (0, 255, 255), 1)
                    #cv2.rectangle(img, (min_x, max_y),(max_x, min_y), (0, 255, 0), 1)
        #print(box_list)
    csv_path = imgpath.split(".")[0]
#     if not os.path.exists(csv_path + ".csv"):
#         #cv2.putText(img, "DIR : " + "images" + " - " + tail, (20, 50),
#         #            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
#         #cv2.imshow('Verifying annotation ', img)
#         save_csv(csv_path + ".csv", csvholder)
#         print("===== saving csv file for ", tail)
#     #cv2.waitKey(2)  # close window when a key press is detected
    return csvholder

get_csv(path+file,path_mat+file_mat)

image_path_array = []
path = "/Users/heronoop/Desktop/FPT/Data/test_dataset/test_data/images/"
dirs = os.listdir(path)
for file in dirs:
    if file.split('.')[1] == 'jpg':
        image_path_array.append(path+file)
image_path_array.sort()
#print(image_path_array)

mat_path_array = []
path_mat = "/Users/heronoop/Desktop/FPT/Data/test_dataset/test_data/annotations/"
dirs_mat = os.listdir(path_mat)
for file_mat in dirs_mat:
    if file_mat.split('.')[1] == 'mat':
        mat_path_array.append(path_mat+file_mat)
mat_path_array.sort()

csvholder = []
for i in range(len(image_path_array)):
    filename = image_path_array[i].split('.')[0].split('/')[-1]
    img = image_path_array[i]
    mat = mat_path_array[i]
    csv_file = get_csv(img,mat)
    csvholder += csv_file

print(len(image_path_array))
print(len(mat_path_array))
print(len(csvholder))
save_csv("/Users/heronoop/Desktop/FPT/Data/test_dataset/test_data/test_labels.csv", csvholder)

# if not os.path.exists(csv_path + ".csv"):
#     save_csv("/Users/heronoop/Desktop/FPT/Data/test_dataset/test_data/test_labels.csv", csvholder)
#     print("===== saving csv file")
#     # cv2.waitKey(2)  # close window when a key press is detected

header = ['filename', 'width', 'height',
          'class', 'xmin', 'ymin', 'xmax', 'ymax']
csv_path = "/Users/heronoop/Desktop/FPT/Data/test_dataset/test_data/test_labels.csv"

csvholder = []
csvholder.append(header)

csv_file = open(csv_path, 'r')
reader = csv.reader(csv_file)

for row in reader:
    csvholder.append(row)
csv_file.close()
        
os.remove("/Users/heronoop/Desktop/FPT/Data/test_dataset/test_data/test_labels.csv")

save_csv("/Users/heronoop/Desktop/FPT/Data/test_dataset/test_data/test_labels.csv", csvholder)
#print("Saved label csv for ", dir, image_dir + dir + "/" + dir + "_labels.csv")
