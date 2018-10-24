import os

import keras as krs
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import numpy as np
from keras import Sequential
# Adding Seed so that random initialization is consistent
from numpy.random import seed

from loader import Loader



class ImageKerasClassifier:

    def __init__(self):
        seed = 1
        np.random.seed(seed)
        self.model_dir = './model'
        self.model_name = 'MNIS'
        self.results = None



    def set_model_dir(self, path):
        self.model_dir = path

    def set_model_name(self, name):
        self.model_name = name

    def get_model_dir(self):
        if self.model_dir.startswith('/'):
            return self.model_dir
        else:
            return './{}'.format(self.model_dir)

    def get_model_name(self):
        return self.model_name

    def get_model_path(self):
        return os.path.abspath('{}/{}'.format(self.get_model_dir(), self.model_name))

    # define the keras model
    def keras_model(self, num_classes):
        # create model
        model = Sequential()
        model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(15, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


    def print_results(self):
        # print results
        if self.results:
            for item in self.results:
                print('--------------------')
                print(item[1])
                i = 0
                for p in item[2]:
                    print("{} = {:.5f}".format(i, float(p)))
                    i += 1
        else:
            print('No results.')

    def print_html_report(self):
        # generate html report page
        file = open("report_{}.html".format(self.model_name), "w")
        file.write("<html><body>")
        file.write("<h2>Results:</h2>")
        file.write("<style>td{padding:15px;border:solid 1px #000}</style>")
        file.write(self.generate_result_table())
        file.write("</body></html>")
        file.close()

    def generate_result_table(self):
        html = ""
        if self.results:
            j = 1
            for item in self.results:
                i = 0
                s = ""
                winner = 0
                number = -1
                ch = item[1].split("\\")[-1][0]
                for p in item[2]:
                    s += "{} = {:.5f}<br/>".format(i, float(p))
                    if float(p) > winner:
                        winner = float(p)
                        number = i

                    i += 1
                s = "<small>{}</small>".format(s)
                path = os.path.abspath(item[1])
                cell2 = "<td>{}<br/><img src='file:///{}' /></td>".format(item[1], path)
                correct = int(ch) == number
                style = '#66FF66' if correct else '#FF6666'
                cell5 = "<td style='background-color:{}'>{}</td>".format(style, correct)
                html += "<tr><td>{}</td>{}<td>{}</td><td><h2>{}</h2></td>{}</tr>".format(j, cell2, s, number, cell5)
                j += 1

        header = "<tr><td>idx</td><td>img</td><td>stat</td><td>pred</td><td>result</td></tr>"
        return "<table style='border:1px'>{}{}</table>".format(header, html)
