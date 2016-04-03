#!/usr/bin/python

import os
import glob
import numpy as np
from PIL import ImageOps
from PIL import Image

data_dir = "/home/rcalland/reactive/handwriting_chinese_100_classes"

img_width = 32
img_height = img_width
num_pixels = img_width * img_height

def convert_images(files):
    images = np.empty((0, num_pixels), np.float32)

    c = 0
    for path in files:
        if (c % 100 == 0):
            print "converting image " + str(c) + " of " + str(len(files))
        c += 1
        img = Image.open(path)
        #resize
        img = img.resize((img_width,img_height))#, Image.ANTIALIAS)
        # invert the grayscale image
        img = ImageOps.invert(img)
        # check that the image is grayscale
        if (img.mode == "L"):
            img_array = np.array(img)
            # 2D to 1D
            img_array = img_array.reshape(img_array.shape[0] * img_array.shape[1])
            # convert from int [0-255], to float [0.0-1.0]
            img_array = img_array.astype(np.float32)
            img_array = np.multiply(img_array, 1.0 / 255.0)
            images = np.append(images, [img_array], axis=0)
        else:
            print "ERROR: images are not grayscale!"
            exit()

    
            
    return images

def convert_labels(labels):
    # find the unique labels
    unique = set(labels)

    # how many classes do we have?
    num_classes = len(unique)
    
    # lets make a dictionary out of this
    onehot_dict = {}

    # now create a one hot vector, probably not the most pythonic method...
    counter = 0
    for label in unique:
        tmp_list = []
        
        for i in xrange(num_classes):
            if (i == counter):
                tmp_list.append(np.uint8(1))
            else:
                tmp_list.append(np.uint8(0))

        counter += 1
                
        onehot_dict[label] = tmp_list

        print onehot_dict

    # now lets assign the onehot labels to each data point
    labels_onehot = np.empty((0, num_classes), np.uint8)

    c = 0
    for lbl in labels:
        if (c % 100 == 0):
            print "converting label " + str(c) + " of "+ str(len(labels))
        c += 1
                      
        labels_onehot = np.append(labels_onehot, [onehot_dict[lbl]], axis = 0)

    return onehot_dict, labels_onehot
        
def find_files(data_dir):
        file_list = []
        label_list = []
        
        for directory in os.listdir(data_dir):
            full_path = os.path.join(data_dir, directory)
            if os.path.isdir(full_path):
                # make a list of all .png files in this directory
                tmp_file_list = glob.glob(full_path + "/*.png")
                file_list += tmp_file_list
                # add the labels
                label_list += [directory for x in tmp_file_list]
                
        return file_list, label_list
                                                                                    
files, labels = find_files(data_dir)
kanji_images = convert_images(files)
onehot_dict, kanji_labels = convert_labels(labels)

np.savez("kanji_data_" + str(img_width) + "x" + str(img_height), kanji_images=kanji_images, kanji_labels=kanji_labels, onehot_dict=onehot_dict)
