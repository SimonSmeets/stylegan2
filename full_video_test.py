import math

import cv2

#tested the use of multiple frames from video and face cropping


def test_video_single(file_name,D):
    face_cascade = cv2.CascadeClassifier('../styleganenv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(file_name)
    results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        new_frame = crop_face(frame, face_cascade).reshape(1, 3, 128, 128)
        results.append(D.run(new_frame, None)[0][0])
        #print("ok")
        break
    cap.release()
    return results[0]



def test_video(file_name,D):
    face_cascade = cv2.CascadeClassifier('../styleganenv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(file_name)
    #dnnlib.tflib.init_tf()
    #G, D, Gs = pickle.load(open(weight, "rb"))
    results = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if count == 1:
            break
        if frame is None:
            break
        new_frame = crop_face(frame, face_cascade).reshape(1, 3, 128, 128)
        results.append(D.run(new_frame, None)[0][0])
        count += 1
        #print("ok")
    print(file_name)
    cap.release()

    return results


def crop_face(image, CascadeClassifier):
    #image = cv2.imread(image_name)
    # Convert the image to gray
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect faces in the image using pre-trained face dectector
    faces = CascadeClassifier.detectMultiScale(gray_image, 1.05, 6)
    # Get the bounding box for each detected face
    i = 6
    while len(faces) == 0 and i >= 0:
        faces = CascadeClassifier.detectMultiScale(gray_image, 1.05, i)
        i -= 1

    if i < 0:
        #print("failed finding face in : " + image_name)
        return cv2.resize(image,(128,128))

    biggestface = [0,0,0,0]
    for face in faces:
        if face[2]*face[3] > biggestface[2]*biggestface[3]:
            biggestface = face

    f = alterbox(biggestface, 1.2)
    x, y, w, h = [v for v in f]
    #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    # Define the region of interest in the image
    cropped = image[y:y + h, x:x + w]
    cropped = cv2.resize(cropped, (128, 128))
    return cropped


def alterbox(face,factor):
    x, y, w, h = face
    w *= factor
    h *= factor

    w = math.ceil(w)
    h = math.ceil(h)

    if w != h:
        w = max(w, h)
        h = max(w, h)


    x -= (factor - 1) * w / 2
    y -= (factor - 1) * h / 2


    x = max(0,math.floor(x))
    y = max(0,math.floor(y))



    return [x, y, w, h]


threshold = -307.81
# #
# test_vid = "../databases/replay-attack/attack_highdef_client001_session01_highdef_photo_adverse.mov"
# real_vid = "../databases/replay-attack/client001_session01_webcam_authenticate_adverse_1.mov"
#
#
# res = test_video(test_vid,"../weights/stylegan2_finetuned_replay/network-snapshot-006677.pkl")
# th_res = [0 if x > threshold else 1 for x in res]
# print(th_res[1:] == th_res[:-1])
#

# tfrecord_dir = "../databases/FFHQ/straight_faces_tf"
# image_dir = '../databases/casia-fasd/split_real/*/*'
# # shuffle = True
# dataset_tool.create_from_images(tfrecord_dir, image_dir, shuffle)