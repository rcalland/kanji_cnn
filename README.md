# Handwritten Kanji Recognition with a CNN
Recognize a set of 100 kanji from handwritten images. The model definition is inside cnn_model.py, the other programs read from this file, so make your edits there only!

#### If you intend to train the model
Please download and unzip this file into the root directory: https://drive.google.com/file/d/0B69xlFeuS9FJN3lycXlfNGFmdWM/view

First you must convert this set of .png files into a preformatted numpy array. This is done with ```python convert_data.py```. This process is a little slow and could be improved. This will also create the labels, and a dictionary allowing us to convert the one-hot vector back to the original kanji label. Make sure to set the location of the unzipped image files in convert_data.py, along with setting the desired image resolution.

To train the model, make any necessary edits to cnn_model.py, then simply run ```python cnn_model.py```

#### To test the model
The repo contains a pre-trained model, so you can try it out by doing
```python evaluate.py image1.png image2.png ... imageN.png```

It will automatically resize the image to whatever the image size is defined as inside cnn_model.py. **It will also plot the image by default, you can change that by setting ```plot_image = False``` inside evaluate.py**

