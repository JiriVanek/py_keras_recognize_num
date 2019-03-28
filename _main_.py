import numpy
from keras import backend as K
import keras
import time
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils

K.set_image_dim_ordering('th')
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))


# define the keras model
def keras_model():
    # create model
    network = Sequential()
    network.add(Conv2D(60, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    network.add(MaxPooling2D(pool_size=(2, 2)))
    network.add(Conv2D(15, (3, 3), activation='relu'))
    network.add(MaxPooling2D(pool_size=(2, 2)))
    network.add(Dropout(0.2))
    network.add(Flatten())
    network.add(Dense(128, activation='relu'))
    network.add(Dropout(0.15))
    network.add(Dense(64, activation='relu'))
    network.add(Dense(num_classes, activation='softmax'))
    # Compile model
    network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return network

def tutorial_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(1, 28, 28)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

allruns=0;
allscore=0;
for x in range(1, 51):
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    start = time.time()
    # build the model
    model = keras_model()
    # model = tutorial_model()
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=200,callbacks=[tbCallBack])
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    end = time.time()
    print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
    allscore+=scores[1]
    run=end-start
    allruns +=run
    print("Avg Time "+ str(allruns/x) + "run " + str(x))
    print("Avg Error: %.2f%%" % (100 - (allscore/x) * 100))
    resultTime=allruns/x






