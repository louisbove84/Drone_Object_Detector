#imports
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import pixellib
from pixellib.tune_bg import alter_bg
import time
import os
import glob
import pandas as pd
import shutil
import io


class Preprocess:
    """ Class used to preprocess images and videos for use in object detection

    This class contains all the methods needed to preprocess images and videos
    for object detection training. Primarily focused on Efficient-Det and Yolov4
    models.

    Methods include:
    1. video_to_frames(input_loc, output_loc)
    2. make_sub_folders(vid_file, output_foldername, classification_name)
    3. create_xml_from_image(output_loc)

    """

    def __init__(self):
        pass

    #Credit: https://stackoverflow.com/users/7319568/harsh-patel
    def video_to_frames(self, input_loc, output_loc):
        """Method to extract frames from input video file
        and save them as separate frames in an output directory.
        
        Args:
            input_loc: Input video file.
            output_loc: Output directory to save the frames.
        
        Returns:
            None
        """
        try:
            os.mkdir(output_loc)
        except OSError:
            pass
        
        # Log the time
        time_start = time.time()
        
        # Start capturing the feed
        cap = cv2.VideoCapture(input_loc)
        
        # Find the number of frames
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        print ("Number of frames: ", video_length)
        count = 0
        print ("Converting video..\n")
        
        # Start converting the video
        while cap.isOpened():
            if ((count%5) == 0):
        
                # Extract the frame
                ret, frame = cap.read()
        
                # Write the results back to output location.
                cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
            count = count + 1
        
            # If there are no more frames left
            if (count > (video_length-1)):
        
                # Log the time again
                time_end = time.time()
        
                # Release the feed
                cap.release()
        
                # Print stats
                print ("Done extracting frames.\n%d frames extracted" % count)
                print ("It took %d seconds forconversion." % (time_end-time_start))
                break
    
    def make_sub_folders(self, vid_file, output_foldername, classification_name):

        """Method to set up folder structure and then calls 
        video_to_frame method to deposite separated frames into 
        file structure used for preprocessing.
        
        Args:
            vid_file: Input video file. 
            output_foldername: Name used for main output folder.
            classification_name: Name to be used for classification purposes.
        
        Returns:
            output_loc: The location of all the separated images
        """
        
        os.mkdir(str(output_foldername))
        os.mkdir(str(output_foldername) + "/training")
        os.mkdir(str(output_foldername) + "/data")
        os.mkdir(str(output_foldername) + "/images")
        os.mkdir(str(output_foldername) + "/images/boxed")
        os.mkdir(str(output_foldername) + "/images/train")
        os.mkdir(str(output_foldername) + "/images/test")
        os.mkdir(str(output_foldername) + "/images/valid")
        input_loc = str(os.getcwd()) + '/' + str(vid_file)
        output_loc = str(os.getcwd()) + '/' + str(output_foldername) + '/images'
        
        #Call method to split video into images folder
        preproc = Preprocess()
        preproc.video_to_frames(input_loc, output_loc)

        return output_loc

    #Credit: 
    # 1. https://www.youtube.com/watch?v=VF8M9DdZ_Aw
    # 2. https://pixellib.readthedocs.io/en/latest/Image_instance.html

    def create_xml_from_image(self, output_loc, classification_name):

        """Method to extract largest identified object in image using 
        FasterRCNN. Method provides both a photo of image with a bounding box
        and a associate XML PASCAL VOC file for every image that contains 
        the bounding box information.
        
        Args:
            output_loc: Folder containing all the images 
        
        Returns:
            None
        """
        
        for filename in os.listdir(output_loc):

            if filename.endswith(".jpg") or filename.endswith(".png"):
                
                #Use FasterRCNN to find the drone and segment it to a white background
                change_bg = alter_bg(model_type = "pb")
                change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
                print(output_loc + filename)
                change_bg.color_bg(output_loc + '/' + filename, colors = (255, 255, 255), output_image_name=output_loc + '/boxed/boxed' +filename)

                #Read in segmented image
                frame = cv2.imread(output_loc + '/boxed/boxed' + filename)

                # Our operations on the frame come here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)
                gray= cv2.medianBlur(gray, 3)   #to remove salt and paper noise

                #Binary threshold
                ret,thresh = cv2.threshold(gray,200,255,0)  

                #Outer boundery only     
                kernel = np.ones((2,2),np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)

                #to strength week pixels
                thresh = cv2.dilate(thresh,kernel,iterations = 5)
                contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                #Draw bounding box
                cv2.drawContours(frame, contours, -1, (0,255,0), 5)

                # find the biggest countour (c) by the area
                c = max(contours, key = cv2.contourArea) if contours else None
                x,y,w,h = cv2.boundingRect(c)

                # draw the biggest contour (c) in blue
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                # Write resulting frame
                cv2.imwrite(output_loc + '/boxed/boxed' +filename,frame)

                #Variables
                xmin = x
                xmax = x + w
                ymin = y
                ymax = y + h
                width, height, depth = frame.shape

                #Create new XML file in PASCAL VOC format 
                with open('box_output.xml', encoding='latin-1') as f:
                    tree = ET.parse(f)
                    root = tree.getroot()

                    #Replace values in XML file
                    for elem in root.getiterator():
                        try:
                            elem.text = elem.text.replace('foldername_', output_loc)
                            elem.text = elem.text.replace('filename_', filename)
                            elem.text = elem.text.replace('width_', str(width))
                            elem.text = elem.text.replace('height_', str(height))
                            elem.text = elem.text.replace('depth_', str(depth))
                            elem.text = elem.text.replace('Drone_name', str(classification_name))
                            elem.text = elem.text.replace('xmin_', str(xmin))
                            elem.text = elem.text.replace('xmax_', str(xmax))
                            elem.text = elem.text.replace('ymin_', str(ymin))
                            elem.text = elem.text.replace('ymax_', str(ymax))

                        except AttributeError:
                            pass

                #Output new XML file
                tree.write(output_loc + '/' + filename[:-4]+'.xml', encoding='latin-1')
        
            else:
                continue
        
        #Allow user to scan through images in the 'boxed' for mistakes and have those files erased
        while True:
            if input("Please take this time to scan through 'images/boxed' folder for images where the bounding box is not correctly identifying the drone. Please delete all misclassified images. Once that is complete please press Enter") != 'y':
                break

        #Remove corresponding jpg/xml files that users deleted from the bounding box images folder
        boxed_files = [str(x)[5:-3] for x in os.listdir(output_loc+'/boxed')]
        print(boxed_files)
        for filename in os.listdir(output_loc):

            if filename.endswith(".xml"):

                if filename[:-3] not in boxed_files:
                    
                    try: 
                        os.remove(output_loc + '/' + filename)
                    except:
                        continue

        #Remove jpg files where bounding boxes were not present
        list_files = [str(x)[:-3] for x in os.listdir(output_loc)]
        single_files = [i for i in list_files if list_files.count(i)==1]
        for file in single_files:
            try: 
                os.remove(output_loc + '/' + file + 'jpg')
            except:
                continue

        #Break the images and XML files into train, test, and valid folders
        jpg_count = 1

        for filename in os.listdir(output_loc):

            if filename.endswith(".jpg") or filename.endswith(".png"):
                if jpg_count % 2 == 0: 
                    shutil.move(output_loc + '/' + str(filename),output_loc + '/train/' + str(filename))
                    shutil.move(output_loc + '/' + str(filename)[:-3] + 'xml',output_loc + '/train/' + str(filename)[:-3] + 'xml')
                    jpg_count += 1
                elif str(jpg_count).endswith('1') or str(jpg_count).endswith('3') or str(jpg_count).endswith('5'):
                    shutil.move(output_loc + '/' + str(filename),output_loc + '/valid/' + str(filename))
                    shutil.move(output_loc + '/' + str(filename)[:-3] + 'xml',output_loc + '/valid/' + str(filename)[:-3] + 'xml')
                    jpg_count += 1                    
                else:
                    shutil.move(output_loc + '/' + str(filename),output_loc + '/test/' + str(filename))
                    shutil.move(output_loc + '/' + str(filename)[:-3] + 'xml',output_loc + '/test/' + str(filename)[:-3] + 'xml')
                    jpg_count += 1

    #Credit: 
    # 1. https://github.com/datitran/raccoon_dataset
    def xml_to_csv(self, output_foldername):
        """Method to translate XML PASCAL VOC files in the train, test, and 
        valid folders into CSV format.
        
        Args:
            None 
        
        Returns:
            None
        """

        for directory in ['train','test','valid']:
            image_path = os.path.join(os.getcwd(), '{}/images/{}'.format(output_foldername, directory))
            data_path = os.path.join(os.getcwd(), '{}/data/{}_labels.csv'.format(output_foldername, directory))
            xml_list = []
            for xml_file in glob.glob(image_path + '/*.xml'):
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for member in root.findall('object'):
                    value = (root.find('filename').text,
                            int(root.find('size')[0].text),
                            int(root.find('size')[1].text),
                            member[0].text,
                            int(member[5][0].text),
                            int(member[5][2].text),
                            int(member[5][1].text),
                            int(member[5][3].text)
                            )
                    xml_list.append(value)
            column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
            xml_df = pd.DataFrame(xml_list, columns=column_name)
            xml_df.to_csv(data_path, index=None)
            print('Successfully converted xml to csv.')
            print("Please open 'generate_tfrecord.py' and add all existing classes to line 35, starting at 1.")


    def make_final_folders(self, final_foldername):

        """Method to set up the final folder structure for preprocessing.
        
        Args:
            final_foldername: Name used for main final folder.
                    
        Returns:
            output_loc: location of the images subfolder
        """

        os.mkdir(str(final_foldername))
        os.mkdir(str(final_foldername) + "/training")
        os.mkdir(str(final_foldername) + "/data")
        os.mkdir(str(final_foldername) + "/images")
        os.mkdir(str(final_foldername) + "/images/train")
        os.mkdir(str(final_foldername) + "/images/test")
        os.mkdir(str(final_foldername) + "/images/valid")
        output_loc = str(os.getcwd()) + '/' + str(final_foldername) + '/images'
        
        return output_loc