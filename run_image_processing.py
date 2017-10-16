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
import image_processing as IP
import glob
from PIL import Image, ImageSequence

'''
What it does: Execute the image processing on the tiff image 
Steps:

- Open a folder with the name of the tiff (eg, folder A of the A.tiff)
- Split the pages of A and make it all images 
- Put all split images in that folder ( eg, A folder has 1.png, 2.png .... 5.png)

- Start image processing on the images one-by-one (eg, now starting on 1.tiff)
- 2, Remove the line
- 3, Blur the images
- 4, Save each steps(1-3) into specific folder for debug (eg, Blur folder) and finally
- save the final images into the processed_images folder

### Call the image_processing.py and loop over the steps above 

Input: image path and save path
Output: save
'''



def run_image_processing_split_frames(img_full_path, file_name, org_path):
    index = IP.split_frames(img_full_path, file_name, org_path)
    return index


def run_image_processing_removed_line(img_path, save_path):
    IP.remove_line(img_path, save_path)


def run_image_processing_blur(img_path, save_path, save_path1):
    IP.blur(img_path, save_path, save_path1)






