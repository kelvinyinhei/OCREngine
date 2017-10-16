
import cv2
import os 
import numpy as np
from scipy.misc import logsumexp
logsumexp(10.0, axis=0, keepdims=True)
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import math
from matplotlib import pyplot as plt
import pylab
import glob
import json
from pathlib import Path


'''
Input: Image of each page of each pdf 
Output: json file , store textline and letters image

what it does: Extract textlines on each cluster
              Extract the letters from each textline
              Update the json 

Steps:

(parallel run)
- loop over the pdf file 
-*1, Detect the exist of json file
-   Run_detect_clusters.py output json file for each page
-2, Check the field of json for each cluster image
-   To a cluster level. 
-   If the "singal" field is 1, it means it finishes clus


Input: which pdf file 
Output: textline images, letters image, update json 


function 1(): vertical projection
Input: image 
Output: return projection result

function 2(): horizontal projection
Input: image 
Output: return projection result 
'''




### test horizontal location



def verticalProjection(img):
    "Return a list containing the sum of the pixels in each column"
    (h, w) = img.shape[:2]
    #print(h,w)
    sumCols = []
    for j in range(w):
        col = img[0:h, j:j+1] # y1:y2, x1:x2
        sumCols.append(np.sum(col))
    result = [x/255 for x in sumCols]
    return result


def horizontalProjection(img):
    "Return a list containing the sum of the pixels in each row"
    (h, w) = img.shape[:2]
    #print(h,w)
    sumRows = []
    for j in range(h):
        row = img[ j:j+1, 0:w]
        sumRows.append(np.sum(row))
    result = [x/255 for x in sumRows]
    return result


def getJson(path):
    with open(path) as json_data:
        data = json.load(json_data)
    return data


def create_letters(data, json_path, img_dir, img_path, clus_name, output_path):
    
    image = cv2.imread(img_path, 0)
    
    (h, w) = image.shape[:2]
    page_no = clus_name.split('_')[0]
    
    plot_y = verticalProjection(image)
    
    mean_y = np.mean(plot_y)
    std_y = np.std(plot_y)
    thres_y = mean_y + std_y/2
    
    

    left_point = []
    right_point = []

    for i in range(w):
        if (len(right_point)) == (len(left_point)):
            if (plot_y[i] < thres_y) and (plot_y[i-1] > thres_y):
                left_point.append(i-1)
                #print('left_point: ', i-1)
                
        if (len(right_point) < len(left_point)):
            if (plot_y[i] > thres_y) and (plot_y[i-1] < thres_y):
                right_point.append(i)
                #print('right_point:', i)
                
    if (len(right_point) < len(left_point)):
        right_point.append(w)

    for n, i in enumerate(left_point):
        if i < 0:
            left_point[n] = 0
    for n, i in enumerate(right_point):
        if i < 0:
            right_point[n] = 0

    
    ### for debug
    letters_contour_path = os.path.join(img_dir, 'processed_images', 'letters_contour')
    if not os.path.exists(letters_contour_path):
        os.makedirs(letters_contour_path)
        
    img_test = cv2.imread(img_path, 3)
    for i in range(len(left_point)):
        cv2.rectangle(img_test, (left_point[i], 0), (right_point[i], 24), (0, 0, 255), 1)
    output_rect = os.path.join(letters_contour_path, clus_name+'_'+'test.png')
    cv2.imwrite(output_rect, img_test)


    
    clus_x = clus_name.split('_')[1]
    clus = clus_name.split('.')[0]
    for i in range(len(left_point)):
        
        letter_name = clus + '_' +  str(int(clus_x) + left_point[i]) + '.png'
        data[page_no][clus_name]['letters'][letter_name] = {}
        data[page_no][clus_name]['letters'][letter_name]['x1'] = str(int(clus_x) + left_point[i])
        data[page_no][clus_name]['letters'][letter_name]['x2'] = str(int(clus_x) + right_point[i])
        data[page_no][clus_name]['letters'][letter_name]['W'] = right_point[i] - left_point[i]

        img = image[0:h, left_point[i]:right_point[i]]
        out_path = os.path.join(output_path, letter_name)
        cv2.imwrite(out_path,img)

    # update json file
    output_json = json_path
    with open(output_json, 'w') as fp:
        json.dump(data, fp)


