
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




### Find the words in each clusters and output as csv file
def getJson(path):
    with open(path) as json_data:
        data = json.load(json_data)
    return data


def combine_letters(json_path, image_number, r_dir):
    
    data = getJson(json_path)
    image_number = image_number


    # Find all the clusters in one single page 
    cluster = []
    page = data[str(image_number)]
    clusters = list(page.keys())


    ### create dataframe for storing the result for outputing as excel file 
    columns = ['filename', 'bbid', 'bbclass', 'word', 'bbCoord_x0', 'bbCoord_y0', 'bbCoord_x1', 'bbCoord_y1', 'x_wconf', 'font_size']


    zero = [0]* (len(columns))
    df = pd.DataFrame(zero).transpose()
    df.columns = columns 
    #print(df)
    filename = os.path.join(r_dir, str(image_number) + '.png')



    num = 1
    bbid = 'word_1_'+str(num)

    bbclass = 'ocrx_word'
    index_num = 0 

    # index on the dataframe 
    times = 0
    # for counting the bbid
    num =1



    for clus in clusters:

        cluss = data[image_number][clus]['letters']
        letters = []

        letters = list(cluss.keys())
        letters = sorted(letters)
        
        
        ### store the y1, y2
        y1 = data[image_number][clus]['y1']
        y2 = data[image_number][clus]['y2']
        ### store the x1, x2, result 
        x1= []
        x2 =[]
        result = []
        for char in letters:
            char_x1 = data[image_number][clus]['letters'][char]['x1']
            x1.append(char_x1)
            ######print(char_x1)
            char_x2 = data[image_number][clus]['letters'][char]['x2']
            x2.append(char_x2)
            ######print(char_x2)
            res = data[image_number][clus]['letters'][char]['result']
            result.append(res)
            ######print(res)


        
        
        
        diff = []
        for i in range(len(x1)-1):
            diff.append(int(x1[i+1])-int(x2[i]))
            
        if len(diff) > 0:
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            thres_diff = mean_diff + std_diff/2
        
        space_loc = []
            
        for i in range(len(diff)):
            if (diff[i] > thres_diff):
                space_loc.append(i)
        

        
        
        space = 0
        whole_words = []
        whole_words_x1 = []
        whole_words_y1 = y1
        whole_words_x2 = []
        whole_words_y2 = y2
        whole_words_H = y2-y1
        
        word = []
        for idx, char in enumerate(result):
            word.append(char)
            # get the first x1 
            if idx == 0:
                whole_words_x1.append(x1[idx])

                
            if (len(space_loc) > 0):
                if idx == space_loc[space]:
                    whole_words_x2.append(x2[idx])
                    words = ''.join(word)
                    whole_words.append(words)
                    word = []
                    
            if (len(space_loc) > 0) :
                if idx == space_loc[space]+1:
                    whole_words_x1.append(x1[idx])
                    if space < len(space_loc)-1:
                        space+=1
                    
            if idx == (len(result)-1):
                whole_words_x2.append(x2[idx])
                words = ''.join(word)
                whole_words.append(words)

        
        filename_new = [filename]*(len(whole_words))
        bbid = []
        for i in range(len(whole_words)):
             res = 'word_1_'+str(num)
             bbid.append(res)
             num+=1
        bbclass_new = [bbclass]*(len(whole_words))
        word = whole_words
        bbCoord_x0 = whole_words_x1
        bbCoord_y0 = [whole_words_y1]*len(whole_words)
        bbCoord_x1 = whole_words_x2
        bbCoord_y1 = [whole_words_y2]*len(whole_words)
        
        x_wconf = [0]*len(whole_words)
        font_size = [whole_words_H]*len(whole_words)

        
        for (a,b,c,d,e,f,g,h,i,j) in zip(filename_new,bbid,bbclass_new,word,bbCoord_x0,bbCoord_y0,bbCoord_x1,bbCoord_y1,x_wconf,font_size):
            res = [a,b,c,d,e,f,g,h,i,j]
            df.loc[times] = res
            times += 1
        
    # re-index by the X0 column
    df = df.sort_values(['bbCoord_y0', 'bbCoord_x0'])
    num = 1
    bbid = []
    for i in range(len(df['bbid'])):
        res = 'word_1_'+str(num)
        bbid.append(res)
        num+=1
    df['bbid'] = bbid
    
    return df
