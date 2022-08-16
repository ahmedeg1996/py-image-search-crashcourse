from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from imutils import paths
import cv2
import argparse
import numpy as np
import os


def image_to_vector(image , size = (32,32)):

    return cv2.resize(image , size).flatten()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model",
	help="path to output model file")
args = vars(ap.parse_args())

# data preprocessing



paths = list(paths.list_images(args["dataset"]))
data = []
labels = []
for (i,imagepath) in enumerate (paths):

    image = cv2.imread(imagepath)

    label  = imagepath.split(os.path.sep)[-1].split(".")[0]

    features = image_to_vector(image)



    data.append(features)
    labels.append(label)
data = np.array(data) / 255.0
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels,2)



(trainData, testData, trainLabels, testLabels) = train_test_split(
	data, labels, test_size=0.25, random_state=42)


model = Sequential()
model.add(Dense(768 , input_dim = 3072 , kernel_initializer = "uniform" , activation = "relu"))
model.add(Dense(384 , kernel_initializer = "uniform", activation = "relu"))
model.add(Dense(2))
model.add(Activation ("softmax") )

sgd = SGD(lr=0.01)

model.compile(loss = "binary_crossentropy" , optimizer = sgd , metrics = ["accuracy"])
model.fit(trainData, trainLabels, epochs=50, batch_size=128, verbose=1)

(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)

print("loss = {:.4f} , accuracy = {:.4f}".format(loss , accuracy*100))

model.save("my model.h5")
