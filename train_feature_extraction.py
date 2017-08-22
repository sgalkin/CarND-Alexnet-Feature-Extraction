import datetime as dt
import pickle
import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

with open('train.p', 'rb') as data:
    train = pickle.load(data)

X_train, X_test, y_train, y_test = train_test_split(
    train['features'], train['labels'], test_size=0.2)

nb_classes = np.unique(np.concatenate((y_train, y_test))).shape[0]

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, nb_classes)

resized = tf.image.resize_images(x, (227, 227))

# NOTE: By setting `feature_extract` to `True` we return the second
# to last layer.
fc7 = AlexNet(resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# NOTE:  use this shape for the weight matrix
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-3))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(
            accuracy_operation, 
            feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

rate = 0.001
EPOCHS = 10
BATCH_SIZE = 32

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train[0:100])
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        epoch_begin = dt.datetime.now()
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(
                training_operation, 
                feed_dict={x: batch_x, y: batch_y})
            
        train_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_test, y_test)
        epoch_end = dt.datetime.now()
        print("E{:02} {:.3f}s => "
              "Train Accuracy = {:.3f}; Validation Accuracy = {:.3f}".
              format(
                  i+1, (epoch_end - epoch_begin).total_seconds(), 
                  train_accuracy, keep_probability, 
                  validation_accuracy))

    saver.save(sess, './alexnet')
    print("Model saved")
