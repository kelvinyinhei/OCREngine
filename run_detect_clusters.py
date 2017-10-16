
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
import glob
import json
from collections import OrderedDict




'''

what it does: MSER/contour to detect clusters on blur images,
              Retrieve each clusters of original image 

Steps:
- becoz it is running independently
- *** first, check the tiff folder if it is exiting
- Then, EG, check 1.png, execute it. Then ckeck for 2.png...
- In each folder, Check if the first page exits in the processed_images file (1.tiff)
- if yes, 
- start run the localizer function one-by-one
- And, save all clusters into the clusters folder 



Input: which tiff file
Output: json file , store cluster image



function 1(): localizer
Input: image path, img_name, save path
Output: save cluster image, output json



One field to the json to 1 when finish MSER to find the clusters
'''






def try_new(img_path, img_name, root_dir):
    
    page_no = img_name.strip('.')[0]
    data = OrderedDict()
    data[page_no] = OrderedDict()
    output_json = os.path.join(img_path, page_no + '.json')


    img = cv2.imread(os.path.join(img_path, img_name), 0)
    
    original_path = os.path.join(root_dir, img_name)
    imgg = cv2.imread(original_path, 0)
    # for debug
    img_test = cv2.imread(original_path, 3)


    ret, thresh = cv2.threshold(img, 100, 200, 0)
    #_, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    prev = [0,0,0,0]
    # thres
    thres = 10
    thr = 3
    
    for cnt in contours:
        
        x, y, w, h = cv2.boundingRect(cnt)

        # for debug   
        cv2.rectangle(img_test, (x-thres,y-thr), (x+w+thres, y+h+thr), (0, 0, 255), 1)

        clus_name = (str(page_no)+"_"+ str(x-thres) + "_" + str(y-thr) + "_" + str(x+w+thres) +"_"+ str(y+h+thr)+".png")
        
        data[page_no][clus_name] = OrderedDict()
        data[page_no][clus_name]['x1'] = x - thres
        data[page_no][clus_name]['y1'] = y - thr
        data[page_no][clus_name]['x2'] = x + w + thres
        data[page_no][clus_name]['y2'] = y + h + thr
        data[page_no][clus_name]['W'] = w
        data[page_no][clus_name]['H'] = h
        data[page_no][clus_name]['letters'] = {}


        
        ### output path
        output = os.path.join(img_path, 'clusters')
        fname = os.path.join(output , clus_name)

        bk_img = imgg[y:y+h+thr, x-thres:x+w+thres]
        cv2.imwrite(fname,bk_img)

    # for debug            
    output_rect = os.path.join(img_path, 'clusters', 'test_contour.png')
    cv2.imwrite(output_rect, img_test)# + '/' + 'test_contour.png', img_test)
    
    with open(output_json, 'w') as fp:
        json.dump(data, fp)




