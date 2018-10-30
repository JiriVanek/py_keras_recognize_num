import numpy
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils

K.set_image_dim_ordering('th')

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


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
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

# build the model
model = keras_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))






