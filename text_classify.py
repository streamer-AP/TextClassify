# Import required modules
import cv2 as cv
import math
import argparse
import numpy as np
import time
parser = argparse.ArgumentParser(description='Use this script to run text detection deep learning networks using OpenCV.')
# Input argument
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
# Model argument
parser.add_argument('--model', default="frozen_east_text_detection.pb",
                    help='Path to a binary .pb file of model contains trained weights.'
                    )
# Width argument
parser.add_argument('--width', type=int, default=320,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.'
                   )
# Height argument
parser.add_argument('--height',type=int, default=320,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.'
                   )
# Confidence threshold
parser.add_argument('--thr',type=float, default=0.999,
                    help='Confidence threshold.'
                   )
args = parser.parse_args()

def classify( input_file,net,confThreshold,inpWidth,inpHeight):
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")
    frame=cv.imread(input_file)

    if frame.ndim==3:
        # Create a 4D blob from frame.
        blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        # Run the model
        net.setInput(blob)
        scores = net.forward(outputLayers)[0]
        if np.max(scores)>confThreshold:
            return True
        else:
            return False
if __name__ == "__main__":
    # Read and store arguments
    confThreshold = args.thr
    inpWidth = args.width
    inpHeight = args.height
    input_file=args.input
    model = args.model
    net = cv.dnn.readNet(model)
    start=time.time()
    print(classify(input_file,net,confThreshold,inpHeight,inpWidth))


