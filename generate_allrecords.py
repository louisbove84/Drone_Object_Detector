import numpy as np
import cv2
import xml.etree.ElementTree as ET
import pixellib
from pixellib.tune_bg import alter_bg
import time
import os
import shutil
from functions import *

if __name__=="__main__":

    #final_foldername = input("Please provide a name for the folder structure for the final model: ")
    
    prepro = Preprocess()

    #Create folder structure and break video file into images
    #output_loc = prepro.make_final_folders(final_foldername)
    output_loc = "/home/louis/Desktop/final_test/images"

    #Allow user to compile all images and XML files into the master folders
    while True:
        if input("Please take this time to take all the images and XML files from the individual train, valid, and test folders and put them in the folders that were just created. Once that is complete please press Enter") != 'y':
            break

    #Translate XML PASCAL VOC files into CSV files
    prepro.xml_to_csv(final_foldername)
            
    #Create TF_Records from the CSV files for use in Efficient-Det Training
    os.system("python generate_tfrecord.py --csv_input={data}train_labels.csv  --output_path={data}train.tfrecord --image_dir={image}/train".format(data = output_loc[:-7]+'/data/', image = output_loc))
    os.system("python generate_tfrecord.py --csv_input={data}test_labels.csv  --output_path={data}test.tfrecord --image_dir={image}/test".format(data = output_loc[:-7]+'/data/', image = output_loc))
    os.system("python generate_tfrecord.py --csv_input={data}valid_labels.csv  --output_path={data}valid.tfrecord --image_dir={image}/valid".format(data = output_loc[:-7]+'/data/', image = output_loc))

    #Create yolo txt files from PASCAL VOC files for use in Yolov5 Training
    os.system("python convert_voc_to_yolo.py --xml_input={image}".format(image = output_loc))
    
