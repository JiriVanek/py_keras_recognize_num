# numbers_recognition
Building a small neural network to recognise hand written numbers using TensorFlow.

A dataset with handwritten numbers was used from MNIST
http://yann.lecun.com/exdb/mnist/

### Installation  
- It uses Python3.5 and keras with GPU support
- [https://www.tensorflow.org](https://www.tensorflow.org)



```
$ python3
Python 3.6.0 (default, Dec 24 2016, 08:01:42)
[GCC 4.2.1 Compatible Apple LLVM 8.0.0 (clang-800.0.42.1)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2 as cv
>>> print(cv.__version__)
3.4.1
```

If it works then install [www.tensorflow.org/install](https://www.tensorflow.org/install/)


## How the project works

All training data are stored in a folder `./data/{dataset_name}/{classes}`
When a training process get started it stores its model in folder `./model/`
Once the model is built than it is used for further prediction. The images
to predict are different from training images. They are stored in
a subfolder `./data/{dataset_name}/predict/` The first letter of a filename
is actually `class name` used to check results. Results are printed on a screen
and an html page with images are created.

