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

    vid_file = input("Which video file would you like to create your dataset from: ")
    #vid_file = 'ghawk.mp4'
    output_foldername = input("Please provide a name for the output folder: ")
    #output_foldername = 'ghawk'
    classification_name = input("What classification name will this video be for: ")
    #classification_name = 'ghawk'
    
    prepro = Preprocess()

    #Create folder structure and break video file into images
    output_loc = prepro.make_sub_folders(vid_file, output_foldername, classification_name)
    #output_loc = '/home/louis/Desktop/reaper_test/images/'
    
    #Create XML files for all images
    prepro.create_xml_from_image(output_loc, classification_name)



