import numpy as np
import tensorflow as tf

data_file_location = "kanji_data_28x28.npz" # train on this data
dict_file_location = "label_dictionary.npz" # where is the label dictionary?
model_save_location = "kanji_cnn_model.ckpt" # save the trained model to this file
n_steps_print = 100 # print training status frequency

# parameters
img_width = 28
img_height = img_width
num_pixels = img_width * img_height
num_classes = 100 # how many kanji are in the set
num_test = 256 # how many entries should be reserved for testing?
batch_size = 50 # batch size for training step
num_steps = 20000 # training steps
keep_probability = 0.5 # dropout probability
conv_filter_size = 5

def load_files():
    # load data from file
    input_file = np.load(data_file_location)
    dict_file = np.load(dict_file_location)
    
    kanji_images = input_file["kanji_images"]
    kanji_labels = input_file["kanji_labels"]
    onehot_dict = dict_file["onehot_dict"].item()
    
    #print onehot_dict
    num_classes = len(onehot_dict)
    
    # remove a random selection of the data to keep for testing
    test_idx = np.random.choice(kanji_images.shape[0], num_test)
    test_images = kanji_images[test_idx]
    test_labels = kanji_labels[test_idx]
    
    kanji_images = np.delete(kanji_images, test_idx, axis=0)
    kanji_labels = np.delete(kanji_labels, test_idx, axis=0)

    return kanji_images, kanji_labels, test_images, test_labels, num_classes
    
def get_batch(b_size, _kanji_images, _kanji_labels):
    idx = np.random.choice(_kanji_images.shape[0], b_size)
    batch_xs = _kanji_images[idx]
    batch_ys = _kanji_labels[idx]
    return batch_xs, batch_ys          

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# placeholders
x = tf.placeholder(tf.float32, shape=[None, num_pixels])
y_ = tf.placeholder(tf.float32, shape=[None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# build the network
W_conv1 = weight_variable([conv_filter_size, conv_filter_size, 1, 32])
b_conv1 = bias_variable([32])

def model(_x, _W_conv1, _b_conv1, _keep_prob):
    x_image = tf.reshape(_x, [-1, img_width, img_height, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, _W_conv1) + _b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([conv_filter_size, conv_filter_size, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    final_size = (img_width / 4) * (img_height / 4)

    W_fc1 = weight_variable([final_size * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, final_size * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, _keep_prob)

    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv

# model
cnn_model = model(x, W_conv1, b_conv1, keep_prob)

# clip the value to avoid NaN
cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(cnn_model, 1E-12, 1.0)) ) 

# use stochastic gradient descent
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
prediction = tf.equal(tf.argmax(cnn_model,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

# init everything
init = tf.initialize_all_variables()

def main(argv=None):
    kanji_images, kanji_labels, test_images, test_labels, num_classes = load_files()
    
    with tf.Session() as sess:
        sess.run(init)
        
        # set up saving 
        saver = tf.train.Saver()
        
        # start the training loop
        for i in range(num_steps):
            # fetch a random selection from the training data
            batch_data, batch_labels = get_batch(batch_size, kanji_images, kanji_labels)

            # print every Nth step
            if (i % n_steps_print == 0):
                train_accuracy = sess.run(accuracy, feed_dict={x: batch_data, y_: batch_labels, keep_prob: 1.0})
                loss = sess.run(cross_entropy, feed_dict={x: batch_data, y_: batch_labels, keep_prob: 1.0})
                print "[" + str(i) + "/" + str(num_steps) + "] - loss for current batch: " + "{:.4f}".format(loss) + ", accuracy: " + "{:.4f}".format(train_accuracy)

            # train
            train_step.run(feed_dict={x: batch_data, y_: batch_labels, keep_prob: keep_probability})

        # finished!    
        print("- Final accuracy %g"%accuracy.eval(feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0}))

        # save trained model
        save_path = saver.save(sess, model_save_location)
        print("- Model saved to: %s" % save_path)

if __name__ == '__main__':
    tf.app.run()
