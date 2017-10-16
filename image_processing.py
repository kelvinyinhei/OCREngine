import cv2
import numpy as np
from matplotlib import pyplot as plt
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

import cv2
import numpy as np
import scipy
from scipy import stats
import math
global debug
from PIL import Image, ImageSequence
import os
import pandas as pd



'''
What it does: Execute the image processing 

function 1(): Remove line

Input: image path and save path (save_path for re-use)
Output: save 

function 2(): Blur the image

Input: image path, save path 
Output: save

function 3(): Align rotation of image

Input: image path, save path 
Output: save

function 4(): Split pdf to images

Input: image path, save path
Output: save 

'''


###     Remove Line
def remove_line(img_path, save_path):
    img = cv2.imread(img_path, 0)

    linek = np.zeros((11,11),dtype=np.uint8)
    linek[5,...] = 1
    x = cv2.morphologyEx(img, cv2.MORPH_OPEN, linek ,iterations=1)
    img -= x
    
    cv2.imwrite(save_path, img)

    img = cv2.imread(save_path)
    black = np.array([0,0,0], dtype = "uint16")
    white = np.array([70,70,70], dtype = "uint16")
    img = cv2.inRange(img, black, white)
    cv2.imwrite(save_path, img)


###     Blur the image
def blur(img_path, save_path, save_path1):
    arr = cv2.imread(img_path, 0)
    horizontal = 5
    vertical = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(horizontal,vertical))
    arr1=cv2.erode(arr, kernel = kernel, iterations=10,anchor=(-1,-1))
    arr2=cv2.dilate(arr1, kernel = kernel, iterations=10,anchor=(-1,-1))
    cv2.imwrite(save_path, arr2)
    
    cv2.imwrite(save_path1, arr2)



def split_frames(img_full_path, file_name, org_path):

    pdf_folder_path = org_path
    if not os.path.exists(pdf_folder_path):
        os.makedirs(pdf_folder_path)

    im = Image.open(img_full_path)
    index = 1
    for frame in ImageSequence.Iterator(im):
        frame = frame.convert('RGB')

        frame_file_path = str(index) +'.png'
        
        img_folder_path = os.path.join(org_path, str(index))
        if not os.path.exists(img_folder_path):
            os.makedirs(img_folder_path)


        blur_path = os.path.join(org_path, str(index), 'blur')
        if not os.path.exists(blur_path):
            os.makedirs(blur_path)


        removed_path = os.path.join(org_path, str(index), 'removed_line')
        if not os.path.exists(removed_path):
            os.makedirs(removed_path)

        processed_image_path = os.path.join(org_path, str(index), 'processed_images')
        if not os.path.exists(processed_image_path):
            os.makedirs(processed_image_path)
    
        
        frame_file_full_path = os.path.join(org_path,frame_file_path)
        frame.save(frame_file_full_path) #save the frames image at org_data
        index += 1
    return index-1



def sharpen_image(img_path):
    img = cv2.imread(img_path)

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(img_path, img)

 
