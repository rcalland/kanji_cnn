# Handwritten Kanji Recognition with a CNN

Recognize a set of 100 kanji from handwritten images. The model definition is inside cnn_model.py, the other programs read from here, so make your edits here only!

#### If you intend to train the model
Please download and unzip this file into the root directory: https://drive.google.com/file/d/0B69xlFeuS9FJN3lycXlfNGFmdWM/view
To train the model, make any necessary edits to cnn_model.py, then simply run python cnn_model.py

#### To test the model
The repo contains a pre-trained model, so you can try it out by doing
```python evaluate.py /path/to/image.png```

It will automatically resize the image to whatever the image size is defined as inside cnn_model.py. It will also plot the image by default, you can change that by setting "plot_image" to False.

