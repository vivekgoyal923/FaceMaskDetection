import detect_and_predict_mask as dpm
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import cv2


def detect_mask():
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    faceNet = cv2.dnn.readNet("model/deploy.prototxt", "model/res10_300x300_ssd_iter_140000.caffemodel")
    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model("model/trained_model")
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    # loop over the frames from the video stream
    # while True:
    #vs = cv2.VideoCapture('model/Mask_Video.mp4')
    #plt.subplots(figsize=(10, 10))
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        #frame = cv2.imread("model/mask_image.jpg")
        frame = imutils.resize(frame, width=400)
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = dpm.detect_and_predict_mask(frame, faceNet, maskNet)
        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask < withoutMask else "No Mask"
            color = (0, 230, 0) if label == "Mask" else (0, 0, 255)
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            # show the output frame
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #plt.imshow(frame)
        #display.clear_output(wait=True)
        #display.display(plt.gcf())
        cv2.imshow("Frames", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()