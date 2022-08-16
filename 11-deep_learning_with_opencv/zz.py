import cv2
import argparse
import numpy as np
import time


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True,
	help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

rows = open(args["labels"]).read().strip().split("\n")
classes = []

for r in rows:

	r = r[r.find(" ")+1:].split(",")[0]

	classes.append(r)


blob = cv2.dnn.blobFromImage(image,1,(24,24) , (104,117,123))

net = cv2.dnn.readNetFromCaffe(args["prototxt"] , args["model"])

net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()

print("network started to train at {:.5} and ended at {:.5}.".format((start ,end)))
