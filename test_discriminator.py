# Load pre-trained network.
# if __name__ == '__main__':
import multiprocessing
import os
import pickle
from multiprocessing import Queue

import PIL
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import resize
from sklearn.metrics import roc_curve, confusion_matrix

import dnnlib
import pretrained_networks
from dnnlib import tflib
from full_video_test import test_video
from training import dataset, misc


def create_fakes(nb, weights):
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
                images.append(os.path.join(newpath, img))
        #     if len(images) == nb:
        #         break
        # if len(images) == nb:
        #     break

    lenght = len(images)
    labels = np.zeros(lenght) if real else np.ones(len(images))
    print(labels.shape)
    return images,labels

def test_NUAA(weights,res):
    dnnlib.tflib.init_tf()
    G, D, Gs = pickle.load(open(weights, "rb"))

    nb_images = 500

    images,labels_neg = get_images(nb_images,True)
    negatives = []
    for image in images:
        image = np.array(Image.open(image))
        image = resize(image, res).reshape(1, 3, res[0],res[1])
        negatives.append(D.run(image, None)[0][0])

    print("done negatives")
    images,labels_pos = get_images(nb_images, False)
    positives = []

    for image in images:
        image = np.array(Image.open(image))
        image = resize(image, res).reshape(1, 3, res[0],res[1])
        positives.append( D.run(image, None)[0][0])


    percentage = 0.2
    split_neg = int(np.floor(percentage*len(labels_neg)))
    split_pos = int(np.floor(percentage*len(labels_pos)))
    y_calc_treshold =  np.append(labels_neg[:split_neg], labels_pos[:split_pos])
    y_pred_treshold =  negatives[:split_neg] + positives[:split_pos]
    y_test = np.append(labels_neg[split_neg:], labels_pos[split_pos:])
    y_test_pred = negatives[split_neg:] + positives[split_pos:]
    calculate_metrics(y_pred_treshold,y_calc_treshold,y_test_pred,y_test)



def calculate_metrics_all_frames(train_predictions,train_true,test_predictions,test_true):

    all_training_predictions = []
    all_training_labels = []
    all_testing_predictions = []
    for i in range(0,len(train_predictions)):
        all_training_predictions.extend(train_predictions[i])
        all_training_labels.extend(([train_true[i]]*len(train_predictions[i])))
    for i in range(0,len(test_predictions)):
        all_testing_predictions.extend(test_predictions[i])

    fpr, tpr, threshold = roc_curve(all_training_labels, all_training_predictions , pos_label=1)

    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print(eer_threshold)
    print(eer)


    y_thresholded = [[0 if x > eer_threshold else 1 for x in vid] for vid in test_predictions]
    y_thresholded = [0 if sum(x) < len(x) * 0.1 else 1 for x in y_thresholded]
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
    return hter



def calculate_metrics(train_predictions,train_true,test_predictions,test_true):
    if max(train_true) > 12:
        train_true = [0 if x > 20 else 1 for x in train_true]
    else:
        train_true = [0 if x in [1,2,9] else 1 for x in train_true]

    true_labels = [x[0][0] for x in test_true ]
    if max(test_true) > 12:
        test_true = [0 if x > 20 else 1 for x in test_true]
    else:
        test_true = [0 if x in [1,2,9] else 1 for x in test_true]

    train_predictions = [abs(x) for x in train_predictions]
    test_predictions = [abs(x) for x in test_predictions]


    fpr, tpr, threshold = roc_curve(train_true, train_predictions, pos_label=1)
    print("fpr = ", list(fpr))
    print("tpr = ", list(tpr))
    print("train_threshold = ", list(threshold))
    print("train_hter = " , [(fpr[i] + (1-tpr[i]))/2 for i in range(0,len(fpr))])

    test_fpr, test_tpr, test_threshold = roc_curve(test_true, test_predictions, pos_label=1)
    print("test_fpr = ", list(test_fpr))
    print("test_tpr = ", list(test_tpr))
    print("test_threshold = ", list(test_threshold))
    print("test_hter = " , [(test_fpr[i] + (1-test_tpr[i]))/2 for i in range(0,len(test_fpr))])

    test_predictions = [-x for x in test_predictions]


    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print(eer_threshold)
    print(eer)
    eer_threshold = -eer_threshold

    y_thresholded = [0 if x > eer_threshold else 1 for x in test_predictions]
    wrongly_classified = []
    all_results = []
    for i in range(0,len(true_labels)):
        if y_thresholded[i] != test_true[i]:
            wrongly_classified.append(true_labels[i])
        all_results.append((test_predictions[i],test_true[i]))
    print("wrongly classified: ",wrongly_classified)
    print("all_results", all_results)

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
    return hter


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

            # if "highdef_photo" in filename:
            #     continue

            if filename.endswith(("avi","mov","jpg","png")):
                image = os.path.join(loc,path)
                all_images.append(image)
                if tag in filename or ("casia" in loc and not filename.startswith(("1","2","HR_1")) ):
                    all_labels.append([[1]])
                else:
                    all_labels.append([[0]])
    return all_images,all_labels


def test_replay(weights, multiple = False, res = (128,128)):


    if type(weights) is not list:
        weights = [weights]

    queue = Queue()
    all_hter = []
    for weight in weights:
        #p = multiprocessing.Process(target=process_weight,args=(weight,train_images,train_labels,test_images,test_labels,res, queue,multiple))
        p = multiprocessing.Process(target=process_weight_tf,args=(weight,"devel_faces_norm_scaled_real_tf","devel_faces_norm_scaled_attack_tf","test_faces_norm_scaled_real_tf","test_faces_norm_scaled_attack_tf",res, queue))
        p.start()
        all_hter.append(queue.get())
        p.join()

    for weight, hter in all_hter:
        print(weight, hter*100)

def test_casia(weights, multiple = False, res = (128,128)):


    if type(weights) is not list:
        weights = [weights]

    queue = Queue()
    all_hter = []
    for weight in weights:
        #p = multiprocessing.Process(target=process_weight,args=(weight,train_images,train_labels,test_images,test_labels,res, queue,multiple))
        p = multiprocessing.Process(target=process_weight_tf,args=(weight,"devel_faces_real_tf","devel_faces_attack_tf","test_faces_real_tf","test_faces_attack_tf",res, queue))

        p.start()
        all_hter.append(queue.get())
        p.join()

    for weight, hter in all_hter:
        print(weight, hter*100)


def process_weight_tf(weight,train_images_real,train_images_attack,test_images_real,test_images_attack,res,queue):
    dnnlib.tflib.init_tf()
    print("running: " + weight)
    G, D, Gs = pickle.load(open(weight, "rb"))

    train_pred = []
    train_lab = []

    train = dataset.load_dataset(data_dir="../databases/casia-fasd/faces", tfrecord_dir="train_faces_tf", max_label_size=1, repeat=False, shuffle_mb=0)
    test = dataset.load_dataset(data_dir="../databases/replay-attack/faces", tfrecord_dir="test_faces_tf", max_label_size=1, repeat=False, shuffle_mb=0)

    for x in range(train._np_labels.shape[0]-1):
        try:
            image,label = train.get_minibatch_np(1)
            image = misc.adjust_dynamic_range(image, [0, 255], [-1, 1])
        except:
            break
        train_pred.append(D.run(image, None)[0][0])
        train_lab.append(label)
    print("done train")

    test_pred = []
    test_lab = []

    for x in range(test._np_labels.shape[0]-1):
        try:
            image,label = test.get_minibatch_np(1)
            image = misc.adjust_dynamic_range(image, [0, 255], [-1, 1])

        except:
            break
        test_pred.append(D.run(image, None)[0][0])
        test_lab.append(label)

    hter = calculate_metrics(train_pred, train_lab, test_pred, test_lab)

    queue.put((weight.split("-")[-1].split(".")[0], hter))


def process_weight(weight,train_images,train_labels,test_images,test_labels,res,queue,video):
    dnnlib.tflib.init_tf()
    print("running: " + weight)
    G, D, Gs = pickle.load(open(weight, "rb"))
    train_pred = []

    if video:
        for image in train_images:
            train_pred.append(test_video(image, D))

        print("done train")

        test_pred = []
        for image in test_images:
            test_pred.append(test_video(image, D))

        hter = calculate_metrics_all_frames(train_pred, train_labels, test_pred, test_labels)
    else:
        for image in train_images:
            image = np.array(Image.open(image))
            image = resize(image, res).reshape(1, 3, res[0], res[1])
            train_pred.append(D.run(image, None)[0][0])


        print("done train")

        test_pred = []
        for image in test_images:
            image = np.array(Image.open(image))
            image = resize(image, res).reshape(1, 3, res[0], res[1])
            test_pred.append(D.run(image, None)[0][0])

        hter = calculate_metrics(train_pred, train_labels, test_pred, test_labels,test_images)

    queue.put((weight.split("-")[-1].split(".")[0],hter))


def generate_images(network_pkl, seeds, truncation_psi, Gs):
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    all_images = []
    for seed_idx, seed in enumerate(seeds):
       #print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        all_images.append(np.array(PIL.Image.fromarray(images[0], 'RGB')))
    return all_images


def normalize_img(img):
    pixels = np.asarray(img)
  #  pixels = pixels.astype('float32')
    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / std
    pixels = 255 * (pixels - np.min(pixels)) / np.ptp(pixels)
    pixels = np.floor(pixels).astype('int16')
    return pixels



def get_threshold_ffhq(weight, nb_images):
    #seeds = random.sample(range(0,100000000), nb_images)

    _G, D, Gs = pretrained_networks.load_networks(weight)

    res = (128, 128)

    loc = r'../databases/straight_faces_threshold_norm_scaled/'
    real_img = [os.path.join(loc,name) for name in os.listdir(loc)]
    #real_img = [ + str(img)[:2].zfill(2) + "000/" + str(img).zfill(5) + ".png" for img in img_numbers]

    real_predictions = []
    for image in real_img:
        image = np.array(Image.open(image))
        image = resize(image, res).reshape(1, 3, res[0], res[1])
        real_predictions.append(D.run(image, None)[0][0])

    print("actual predicted: ", len(real_predictions))

    compare_seeds = [x for x in range(0, len(real_predictions))]
    generated_img = generate_images(weight, compare_seeds, None, Gs)
    generated_img = [normalize_img(img) for img in generated_img]

    generated_predictions = []
    for image in generated_img:
        image = resize(image, res).reshape(1, 3, res[0], res[1])
        generated_predictions.append(D.run(image, None)[0][0])

    print("done generated")

    generated_labels = [1 for x in generated_predictions[:len(real_predictions)]]
    real_labels = [0 for x in real_predictions]

    all_pred = generated_predictions[:len(real_predictions)]
    all_labels = generated_labels
    all_pred.extend(real_predictions)
    all_labels.extend(real_labels)

    fpr, tpr, threshold = roc_curve(all_labels, all_pred, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print(eer_threshold)
    print(eer)
    return eer_threshold, D

def test_ffhq(weight,test_images,test_labels,queue,queue_casia):
    res = (128,128)

    threshold, D = get_threshold_ffhq(weight, 2000)
    print("got threshold ffhq")

    test_pred = []
    for image in test_images[0]:
        image = np.array(Image.open(image))
        image = resize(image, res).reshape(1, 3, res[0], res[1])
        #print(D.run(image, None)[0][0])
        test_pred.append(D.run(image, None)[0][0])

    print("done Replay")

    test_pred_casia = []
    for image in test_images[1]:
        image = np.array(Image.open(image))
        image = resize(image, res).reshape(1, 3, res[0], res[1])
        #print(D.run(image, None)[0][0])
        test_pred_casia.append(D.run(image, None)[0][0])

    process_values(weight,test_pred,test_labels[0],threshold,queue)
    process_values(weight,test_pred_casia,test_labels[1],threshold,queue_casia)




def process_values(weight,test_pred,test_labels, threshold, queue):

    test_thresholded = [0 if x > threshold else 1 for x in test_pred]
    conf = confusion_matrix(test_labels ,test_thresholded)
    print(conf)
    TN = conf[0][0]
    FN = conf[1][0]
    TP = conf[1][1]
    FP = conf[0][1]

    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    hter = (FPR + FNR)/2
    print("FPR: ")
    print(FPR)
    print("FNR: ")
    print(FNR)
    print("HTER: ")
    print(hter)
    queue.put((weight.split("-")[-1].split(".")[0], hter))

def test_with_ffhq_threshold(weights):
    if type(weights) is not list:
        weights = [weights]

    queue = Queue()
    queue_casia = Queue()
    all_hter = []
    all_hter_casia = []


    for weight in weights:
        print("processing weight: ", weight)

        p = multiprocessing.Process(target=test_ffhq,args=(weight,test_images,test_labels,queue,queue_casia))
        p.start()
        all_hter.append(queue.get())
        all_hter_casia.append(queue_casia.get())
        p.join()


    print("weights", "Replay", "Casia")
    for i in range(0,len(all_hter)):
        print(all_hter[i][0], all_hter[i][1]*100, all_hter_casia[i][1]*100)





#weights = r'../weights/karras2019stylegan-ffhq-1024x1024.pkl'
#weights = r'../weights/network-snapshot-018513.pkl'
#weights = "../weights/stylegan2-ffhq-config-f.pkl"
#path = "../weights/finetuned_casia_weights"
#weights = "../weights/SGfinetuned/network-snapshot-019053.pkl"
#path = "../weights/stylegan2_normalized_straight"
#path = "../results/stylegan2_training_weights"
#
    # weights = "../weights/stylegan2_training_weights/network-snapshot-012513.pkl"

# #weights = list(filter(lambda x: "-02" in x, weights))
# weights = weights[15:]
## #weights = weights[-1]
# #test_NUAA(weights,(128,128))
#
# #test_with_ffhq_threshold(weights)
#
if __name__ == '__main__':
    path = "../weights/stylegan2_straight_faces/"
    weights = os.listdir(path)
    weights = [os.path.join(path, x) for x in weights if x.endswith(".pkl")]
    weights = sorted(weights)[-1]
    test_replay(weights, False, (128, 128))
    # test_casia(weights, False, (128, 128))