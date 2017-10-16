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
import json
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import shutil



import image_processing as IP
import run_image_processing as run_image
import run_detect_clusters 
import run_localize_letters
import run_recognize_letters
import run_combine_letters
from tkinter import Tk
from tkinter.filedialog import askopenfilename



def run_OCR(input_path, out_dir):

    
    img_full_path = input_path
    image = os.path.split(img_full_path)[1]

    file_name = os.path.splitext(image)[0]
    # tiff image file only 
    if (os.path.splitext(image)[1] != '.tiff'):
        print('Please select ".tiff" file!')
        exit()
  
    org_path = os.path.join(os.path.split(img_full_path)[0],file_name)
    # get the number of page 
    index = run_image.run_image_processing_split_frames(img_full_path, file_name, org_path)

    # open a dataframe for storing result of each page
    columns = ['filename', 'bbid', 'bbclass', 'word', 'bbCoord_x0', 'bbCoord_y0', 'bbCoord_x1', 'bbCoord_y1', 'x_wconf', 'font_size']
    zero = [0]* (len(columns))
    df = pd.DataFrame(zero).transpose()
    df.columns = columns 

    
    for idx in range(1, index+1):
        
        image = str(idx)+'.png'
        img_folder_path = os.path.join(org_path, str(idx))
        root_dir = org_path
        image_number = idx
        
        ###                 Calling the image_processing.py
        
        ###     Call the run_image_processing.py
        # Remove line first
        img_path = os.path.join(root_dir,  image)
        save_path = os.path.join(root_dir, str(image_number), 'removed_line', image)
        #IP.remove_line(img_path, save_path)

        run_image.run_image_processing_removed_line(img_path, save_path)

        # Blur the image with removed line 
        img_path = os.path.join(root_dir, str(image_number), 'removed_line', image)
        save_path = os.path.join(root_dir, str(image_number), 'blur', image) 
        save_path1 = os.path.join(root_dir, str(image_number), 'processed_images', image)

        run_image.run_image_processing_blur(img_path, save_path, save_path1)

        # create the folder for clusters and letters 
        cluster_path = os.path.join(root_dir,str(image_number),'processed_images','clusters')
        if not os.path.exists(cluster_path):
            os.makedirs(cluster_path)

        letter_path = os.path.join(root_dir,str(image_number),'processed_images','letters')
        if not os.path.exists(letter_path):
            os.makedirs(letter_path)

        ###                 localize the clusters regions on the processed image

        ###     detect the clusters regions on the processed image
        img_path = os.path.join(root_dir, str(image_number), 'processed_images')
        original_path = root_dir
        run_detect_clusters.try_new(img_path, image, original_path)

        ###                 localize the letters image 

        json_path = os.path.join(root_dir, str(idx), 'processed_images', str(idx)+'.json')
        # check the json file if exits
        my_file = Path(json_path)
        if not my_file.exists():
            print('Find no json file of image page ', str(idx))

        # Reload the json file 
        data = run_localize_letters.getJson(json_path)

        # Find all clusters in json file for one image page
        clusters = []
        page = data[str(image_number)]
        for name, dict_ in page.items():
            clusters.append(name)

        img_dir = os.path.join(root_dir, str(image_number))
        # Call the run_localize_letters.py
        for clus in clusters:
            img_path = os.path.join(root_dir, str(idx), 'processed_images','clusters', clus)
            clus_name = clus
            output_path = os.path.join(root_dir, str(idx), 'processed_images', 'letters')

            run_localize_letters.create_letters(data, json_path, img_dir, img_path, clus_name, output_path)

        ###                 Letters Recognition
            
        json_path = os.path.join(root_dir, str(idx), 'processed_images', str(idx)+'.json')
        # Reload and update the json file 
        data = run_localize_letters.getJson(json_path)

        # Find all clusters in json file for one single page
        clusters = []
        page = data[str(image_number)]
        for name, dict_ in page.items():
            clusters.append(name)


        # Find all letters in each cluster for one single page 
        letters = []
        roott_dir = os.path.join(root_dir, str(idx), 'processed_images', 'letters' )
        
        for clus in clusters:
            cluss = data[str(image_number)][clus]['letters']
            for name, dict_ in cluss.items():
                letters.append(name)
                    
        
        # Predict the result 
        pre = run_recognize_letters.Predict_Image(roott_dir, letters)


        for index, img_name in enumerate(letters):
            clus = ('_').join(img_name.split('_')[:5])+'.png'
            
            data[str(image_number)][clus]['letters'][img_name]['result'] = {}
            result = pre[index]
            data[str(image_number)][clus]['letters'][img_name]['result'] = result
            
            
         
        # update json file
        output_json = os.path.join(root_dir, str(idx), 'processed_images', str(idx)+'.json')
        with open(output_json, 'w') as fp:
            json.dump(data, fp)


        ###             Combine the letters and output the excel 
        json_path = os.path.join(root_dir, str(idx), 'processed_images', str(idx)+'.json')
        r_dir = os.path.split(img_full_path)[0]
        image_number = str(image_number)
        dff = run_combine_letters.combine_letters(json_path, image_number, r_dir)
        df = df.append(dff, ignore_index = True)


    ### combine and output txt file
    df = df.drop(df.index[0])
    output_path = os.path.join(out_dir)
    df.to_csv(output_path, sep=',', index = None)


    # Remove the existing folder 
    d_filee = os.path.split(img_full_path)[1]
    d_file = os.path.splitext(d_filee)[0]
    delete_path = os.path.split(img_full_path)[0]
    shutil.rmtree(os.path.join(delete_path, d_file))



# Debug

input_path = '/media/SSD/SelfBuildOCR/All_Data/test_flask/A.tiff'
out_dir = '/media/SSD/SelfBuildOCR/All_Data/test_flask/A.txt'
run_OCR(input_path, out_dir)


