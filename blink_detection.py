# import the necessary packages
import math

import os

import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear






def check_blink(file_name):
    cap = cv2.VideoCapture(file_name)
    if not cap.isOpened():
        print("Problem opening video file: " + file_name)
        return -1;

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


    cur_max = 0
    cur_min = 1

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < cur_min:
                cur_min = ear
            if ear > cur_max:
                cur_max = ear
            # print(cur_max - cur_min)


            if (cur_max - cur_min > 0.108): #cas =  0.1365
                return True


    return False

def calc_eye(eye,frame,rad = None):
    center = tuple([math.floor(sum(i) / len(i)) for i in zip(*eye)])
    if rad is None:
        rad = math.floor((max(eye, key = lambda k: k[0])[0] - min(eye, key = lambda k: k[0])[0])/2)

    bottomLeft = (center[0] - rad, center[1] - rad)
    topRight = (center[0] + rad, center[1] + rad)

    lefteyeimage = frame[bottomLeft[1]:topRight[1], bottomLeft[0]:topRight[0]]
    lefteyeimage = cv2.adaptiveThreshold(lefteyeimage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return lefteyeimage,rad


def get_all_images(loc):
    all_images = []
    print(loc)
    for path in os.listdir(loc):
        if os.path.isdir(os.path.join(loc,path)):
            images = get_all_images(os.path.join(loc,path))
            all_images.extend(images)
        else:
            filename = os.fsdecode(path)

            if filename.endswith(("avi","mov","jpg","png")):
                image = os.path.join(loc,path)
                all_images.append(image)

    return all_images


if __name__ == '__main__':
    path = "../databases/replay-attack/train"
    res = []
    all_images = get_all_images(path)
    for img in all_images:
        if "video" not in img:
            print(img)
            result = check_blink(img)
            print(result)
            res.append((img,result))
    print(res)

    # for subdir in sorted(os.listdir(path)):
    #     print(subdir)
    #     dir = os.path.join(path,subdir)
    #     for file in sorted(os.listdir(dir)):
    #         print(file)
    #         if file.rstrip(".avi") in ["1","2","HR_1", "3", "4", "HR_2", "5" , "6" , "HR_3"]:
    #             result = check_blink(os.path.join(dir,file))
    #             print(result)
    #             res.append((file,result))
    print(res)


