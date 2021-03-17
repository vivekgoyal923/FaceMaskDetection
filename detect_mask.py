import detect_and_predict_mask as dpm
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import cv2
from datetime import datetime
import detect_similar_images as dsi
from scipy.spatial.distance import cosine
import numpy as np
import extract_images as ei
from keras_vggface.vggface import VGGFace
import os
import math


face_detect_prototxt = "model/deploy.prototxt"
face_detect_caffemodel = "model/res10_300x300_ssd_iter_140000.caffemodel"
mask_detect_model = "model/trained_model"
DetectedFolder = "DetectedFaces/"
TestImage = "TestImage/"

def detect_mask():
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    faceNet = cv2.dnn.readNet(face_detect_prototxt, face_detect_caffemodel)
    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(mask_detect_model)
    resNet = VGGFace(model='resnet50',
                    input_shape=(224, 224, 3),
                    pooling='avg')
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    #vs = cv2.VideoCapture('model/Mask_Video.mp4')
    #frame = cv2.imread("model/test_image1.jpg")
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (locs, preds, scores) = dpm.detect_and_predict_mask(frame, faceNet, maskNet, resNet)
        for (box, pred, score) in zip(locs, preds, scores):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask < withoutMask else "No Mask"
            color = (0, 230, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            croppedImage = frame[startY:endY, startX:endX]
            if mask > withoutMask:
                new_time = datetime.now()
                old_time = ""
                count = 0
                listOfTestPics = os.listdir(TestImage)
                flag = 0
                for testImage in listOfTestPics:
                    imageToCompare = cv2.imread(TestImage + testImage)
                    frame_to_compare = imutils.resize(imageToCompare, width=400)
                    model_score2 = ei.extract_face_from_image(frame_to_compare, faceNet, resNet)
                    print(cosine(score, model_score2))
                    if cosine(score, model_score2) <= 0.000007:
                        flag = 1
                        name = testImage.split(".")[0]
                        listOfPics = os.listdir(DetectedFolder)
                        time_dif = math.inf
                        for pic in listOfPics:
                            if name in pic:
                                old_time = pic.split(" ")[1]
                                count = int(pic.split(" ")[2][:-4])
                                temp = int(new_time.strftime('%y%m%d%H%M%S')) - int(old_time)
                                if temp < time_dif:
                                    time_dif = temp
                        print(old_time, new_time)
                        if 500 < time_dif < 1000:
                            print("Update on Dashboard")
                            if os.path.exists(f"{DetectedFolder}{name} {old_time} {count}.jpg"):
                                os.remove(f"{DetectedFolder}{name} {old_time} {count}.jpg")
                            count += 1
                            cv2.imwrite(f"{DetectedFolder}{name} {new_time.strftime('%y%m%d%H%M%S')} {count}.jpg", croppedImage)
                        elif time_dif > 1000:
                            print("New Image Uploaded in Detected Folder")
                            if os.path.exists(f"{DetectedFolder}{name} {old_time} {count}.jpg"):
                                os.remove(f"{DetectedFolder}{name} {old_time} {count}.jpg")
                            cv2.imwrite(f"{DetectedFolder}{name} {new_time.strftime('%y%m%d%H%M%S')} {count}.jpg", croppedImage)
                        else:
                            print("Waiting for 2 min")
                        break
                if flag == 0:
                    new_name = input("Please Enter the name for new Face:")
                    if new_name != "ignore":
                        cv2.imwrite(f"{TestImage}{new_name}.jpg", frame)
        cv2.imshow("Frames", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    vs.stop()