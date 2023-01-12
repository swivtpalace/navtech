# CONFIDENTIAL 
# This codes are is intended only for the use of the individual or entity to which it is addressed. It is a classified information that is privileged and confidential.

import glob
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image as k_img
from keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D
from imutils import paths
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import keras
import pickle
from sklearn.model_selection import train_test_split
import cv2
#from keras.preprocessing.image import img_to_array
from keras.utils import img_to_array
import numpy as np
import pandas as pd
import os
import json


TRAIN_PATH = 'photos'


#'/root/jFiles/NT_JWL_Model/img_folder'
def get_loc(train_dir, is_not_test_mode=True):
    train_path = os.path.join('photos', train_dir)  #NT_JWL_Model/img_folder did not give the desired output so added /root/jFiles/

    noi = 0

    image_paths = glob.glob(train_path + '/*')

    noi = len(image_paths)
    global result
    
    tmp = []

    for img in image_paths: 
        if img.endswith(('xlsx', 'png', 'ipynb_checkpoints')):
            tmp.append(img)
    image_paths = list(set(image_paths)-set(tmp))

    noi = len(image_paths)

    multi_model = keras.models.load_model('aimodels/jewel_multitask_model.h5') 

    mlb_cat = pickle.loads(open("aimodels/label_cat.p", "rb").read()) 
    mlb_sty = pickle.loads(open("aimodels/label_sty.p", "rb").read()) 
    mlb_dia = pickle.loads(open("aimodels/label_dia.p", "rb").read()) 

    excel_file = "aimodels/newdatafile2.xlsx"

    xl = pd.ExcelFile(excel_file)
    df = xl.parse(xl.sheet_names[0])

    def pred_data(df, image_paths, model, mlb_cat, mlb_sty, mlb_dia): 

        global  result

        df["pred_category"] = np.nan
        df["pred_style"] = np.nan
        df["pred_shape"] = np.nan
        i = 0
        nm = [None] * noi
        ct = [None] * noi
        st = [None] * noi
        dc = [None] * noi
        ind = [None] * noi

        for imagePath in image_paths: 
            image = cv2.imread(imagePath, -1)

            if len(image.shape) > 2: 
                if image.shape[2]==4: 
                    png = k_img.load_img(imagePath, color_mode='rgba', target_size=(IMAGE_DIMS[1], IMAGE_DIMS[0]), interpolation='bicubic')
                    color=(255, 255, 255)
                    background = Image.new('RGB', png.size, color)
                    background.paste(png, mask=png.split()[3]) 
                    image = k_img.img_to_array(background).astype(int)
                    image = image[:,:,::-1]
                else: 
                    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]), interpolation = cv2.INTER_CUBIC)
                    image = img_to_array(image)                     
            else: 
                image = cv2.imread(imagePath)
                image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]), interpolation = cv2.INTER_CUBIC)

            mean = [103.939, 116.779, 123.68]
            image[..., 0] = image[..., 0] - mean[0]
            image[..., 1] = image[..., 1] - mean[1]
            image[..., 2] = image[..., 2] - mean[2]
            image = np.expand_dims(image, axis=0)

            proba = model.predict(image)

            image_name = imagePath.split(os.path.sep)[-1]

            prob = proba[0].tolist()[0]
            idxs = np.argsort(prob)[::-1][0]
            label_c = mlb_cat.classes_[idxs]

            prob = proba[1].tolist()[0]
            idxs = np.argsort(prob)[::-1][0]
            label_s = mlb_sty.classes_[idxs]

            prob = proba[2].tolist()[0]
            idxs = np.argsort(prob)[::-1][0]
            label_d = mlb_dia.classes_[idxs]

            nm [i] = image_name
            ct [i] = label_c
            st [i] = label_s
            dc [i] = label_d
            ind [i] = i

            df = pd.DataFrame({'Name':(nm),'Category':(ct),'Style':(st),'Cut':(dc)},index=ind)
            result = df.to_json(orient="split")
 
            i = i+1
            #print (i)

    # Ref 1.5

    EPOCHS = 20
    BS = 56
    IMAGE_DIMS = (224, 224, 3)
    test_prop = 0.2

    if noi > 0:
        df = pred_data(df, image_paths, multi_model, mlb_cat, mlb_sty, mlb_dia)
        parsed = json.loads(result)
    else:
        t_ver = "Please upload a file"
        parsed = json.dumps(t_ver)

    pd.set_option("max_rows", None)

    #for removing all the files from the /root/jFiles/NT_JWL_Model/img_folder

    #if is_not_test_mode:
    #    if os.path.exists(train_path) and os.path.isdir(train_path):
    #        os.system(f'rm -rf {train_path}')

    return parsed
