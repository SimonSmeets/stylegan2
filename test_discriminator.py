# Load pre-trained network.
import dnnlib
from dnnlib import tflib
import numpy as np
import pickle

import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize
import os
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix

def create_fakes(nb,weights):
    dnnlib.tflib.init_tf()

    G, D, Gs = pickle.load(open(weights, "rb"))
    # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
    # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
    # Gs = Long-term average of the generator. Yields  higher-quality results than the instantaneous snapshot.
    # Pick latent vector.
    for i in range(0, 10):
        rnd = np.random.RandomState()
        latents = rnd.randn(1, Gs.input_shape[1])

        # # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt).reshape(128, 128,
                                                                                                               3)
        plt.imshow(images)
        plt.show()
        images = images.reshape(1, 1, 3, 128, 128)
        print(images.shape)
        print(D.run(images[0], None))
    D.print_layers()
    return images

def get_images(nb, real):
    if real:
        path = r'../databases/NUAA/Detectedface/ClientFace'
    else:
        path = r'../databases/NUAA/Detectedface/ImposterFace'

    images = []
    for id in os.listdir(path):
        newpath = os.path.join(path,id)
        for img in os.listdir(newpath):
            filename = os.fsdecode(img)
            if filename.endswith("jpg"):
                image = np.array(Image.open(os.path.join(newpath, img)))
                image = resize(image, (128, 128)).reshape(1, 3, 128, 128)
                images.append(image)
        #     if len(images) == nb:
        #         break
        # if len(images) == nb:
        #     break

    lenght = len(images)
    labels = np.zeros(lenght) if real else np.ones(len(images))
    print(labels.shape)
    return images,labels

def test_NUAA(weights):
    dnnlib.tflib.init_tf()
    G, D, Gs = pickle.load(open(weights, "rb"))

    nb_images = 500

    images,labels_neg = get_images(nb_images,True)
    negatives = []
    for image in images:
        negatives.append(D.run(image, None)[0][0])

    print("done negatives")
    images,labels_pos = get_images(nb_images, False)
    positives = []

    for image in images:
        positives.append( D.run(image, None)[0][0])


    percentage = 0.2
    split_neg = int(np.floor(percentage*len(labels_neg)))
    split_pos = int(np.floor(percentage*len(labels_pos)))
    y_calc_treshold =  np.append(labels_neg[:split_neg], labels_pos[:split_pos])
    y_pred_treshold =  negatives[:split_neg] + positives[:split_pos]
    y_test = np.append(labels_neg[split_neg:], labels_pos[split_pos:])
    y_test_pred = negatives[split_neg:] + positives[split_pos:]
    calculate_metrics(y_pred_treshold,y_calc_treshold,y_test_pred,y_test)





def calculate_metrics(train_predictions,train_true,test_predictions,test_true):
    fpr, tpr, threshold = roc_curve(train_true, train_predictions, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print(eer_threshold)
    print(eer)


    y_thresholded = [0 if x > eer_threshold else 1 for x in test_predictions]
    conf = confusion_matrix(test_true,y_thresholded)
    print(conf)
    TN = conf[0][0]
    FN = conf[1][0]
    TP = conf[1][1]
    FP = conf[0][1]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    hter = (FPR + FNR)/2
    print("FPR: ")
    print(FPR)
    print("FNR: ")
    print(FNR)
    print("HTER: ")
    print(hter)


def get_all_images(loc,tag):
    all_images = []
    all_labels = []
    print(loc)
    for path in os.listdir(loc):
        if os.path.isdir(os.path.join(loc,path)):
            images,labels = get_all_images(os.path.join(loc,path),tag)
            all_images.extend(images)
            all_labels.extend(labels)
        else:
            filename = os.fsdecode(path)
            if filename.endswith("jpg"):
                image = np.array(Image.open(os.path.join(loc,path)))
                #image = resize(image, (128, 128)).reshape(1,3, 128, 128)
                image = resize(image, (1024, 1024)).reshape(1, 3, 1024, 1024)
                all_images.append(image)
                if tag in filename:
                #if not filename.startswith(("1","2","HR_1")):
                    all_labels.append(1)
                else:
                    all_labels.append(0)
    return all_images,all_labels

def test_replay(weights, multiple = False):
    dnnlib.tflib.init_tf()
    G, D, Gs = pickle.load(open(weights, "rb"))
    if multiple:
        train_images, train_labels = get_all_images(r'../databases/replay-attack/train_frames_multiple',"attack")
        test_images, test_labels = get_all_images(r'../databases/replay-attack/test_frames_multiple',"attack")
    else:
        train_images, train_labels = get_all_images(
            r'../databases/replay-attack/train_frames',
            "attack")
        test_images, test_labels = get_all_images(
            r'../databases/replay-attack/test_frames',
            "attack")

    print("nb train images: " + str(len(train_labels)))
    print("nb test images: " + str(len(test_labels)))


    train_pred = []
    for image in train_images:
        train_pred.append(D.run(image, None)[0][0])

    print("done train")

    test_pred = []
    for image in test_images:
        test_pred.append(D.run(image, None)[0][0])

    calculate_metrics(train_pred,train_labels,test_pred,test_labels)


def test_casia(weights, multiple = False):
    dnnlib.tflib.init_tf()

    G, D, Gs = pickle.load(open(weights, "rb"))

    if multiple:
        train_images, train_labels = get_all_images(r'../databases/casia-fasd/train_frames_multiple',"attack")
        test_images, test_labels = get_all_images(r'../databases/casia-fasd/test_frames_multiple',"attack")
    else:
        train_images, train_labels = get_all_images(
            r'../databases/casia-fasd/train_frames', "attack")
        test_images, test_labels = get_all_images(
            r'../databases/casia-fasd/test_frames', "attack")

    print("nb train images: " + str(len(train_labels)))
    print("nb test images: " + str(len(test_labels)))


    train_pred = []
    for image in train_images:
        train_pred.append(D.run(image, None)[0][0])

    print("done train")

    test_pred = []
    for image in test_images:
        test_pred.append(D.run(image, None)[0][0])

    calculate_metrics(train_pred,train_labels,test_pred,test_labels)

#weights = r'../results/karras2019stylegan-ffhq-1024x1024.pkl'
#weights = r'../results/network-snapshot-018513.pkl'
weights = "../weights/stylegan2-ffhq-config-f.pkl"


test_NUAA(weights)
#test_replay(weights,False)
#test_casia()