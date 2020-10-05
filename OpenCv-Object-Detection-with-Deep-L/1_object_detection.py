from utils import *
import numpy as np
import cv2
import sys

# help to set image path directory
data_file = sys.argv[1]
file_type = None


# load the caffe model
model_name = 'MobileNetSSD_deploy.caffemodel' # pre-trained model
model_proto = 'MobileNetSSD_deploy.prototxt.txt' # architecture of NN

# Load inside python application
net = cv2.dnn.readNetFromCaffe(model_proto, model_name)

img = cv2.imread(data_file)
# function from utils.py use to draw the box
detect_objects_and_draw_boxes(net, img)

cv2.imshow("Object Detector", cv2.resize(img, (600, 250)))
cv2.waitKey(0)


        