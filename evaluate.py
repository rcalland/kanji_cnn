import numpy as np
from numpy import loadtxt
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import cnn_model
import convert_data

if (len(sys.argv) is 1):
    print "Usage: \"python " + str(sys.argv[0]) + " image1.png image2.png ... imageN.png \""
    quit()

# do you want the input image to be plotted too?
plot_image = True

# file list from arguments
file_list = sys.argv[1:]

print file_list

#quit()
# the model definition is loaded from cnn_model.py
dict_file_location = cnn_model.dict_file_location
model_save_location = cnn_model.model_save_location
labels_unicode_location = "labels_unicode.txt"

# parameters
img_width = cnn_model.img_width
img_height = cnn_model.img_width
num_pixels = img_width * img_height

# fetch unicode characters from file
def get_unicode_labels(filename):
    key_value = loadtxt(filename, delimiter=" ", dtype=np.str)
    mydict = { k:v for k,v in key_value }
    return mydict

# load data from file
imgs = []
for f in file_list:
    _images = convert_data.convert_image(f, img_width, img_height)
    imgs.append(_images)

kanji_images = np.concatenate(imgs, axis=0)

# load the dict file
dict_file = np.load(dict_file_location)
onehot_dict = dict_file["onehot_dict"].item()

#print onehot_dict
num_classes = len(onehot_dict)

# fetch the dictionary to map the labels to unicode characters
label_dict = get_unicode_labels(labels_unicode_location)

# load model
cnnmodel = cnn_model.cnn_model

# set up loading
saver = tf.train.Saver()

char = 0

# reshape array to be able to feed it in
kanji_images = np.reshape(kanji_images, [-1, num_pixels])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    # Load trained model from file
    saver.restore(sess, model_save_location)

    # loop over inputs
    for i in range(kanji_images.shape[0]):
    
        # get the most likely label for input 
        predicted = sess.run(tf.argmax(cnnmodel, 1), feed_dict={cnn_model.x: [kanji_images[char]], cnn_model.keep_prob: 1.0})
        print "- Most likely character for " + str(sys.argv[i+1]) + " is: " + label_dict[onehot_dict.keys()[predicted]].decode('utf-8')

        if (plot_image is True):
            # lets plot the image!
            newimg = np.matrix(kanji_images[char])
            
            # reform a numpy array of the original shape
            arr2 = np.asarray(newimg).reshape((img_width,img_height))
            fig, ax = plt.subplots()
            ax.imshow(arr2,cmap=cm.Greys, interpolation='nearest')
            print "Close image window to continue..."
            plt.show()
