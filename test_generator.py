import multiprocessing
from multiprocessing import Queue
from keras.applications.resnet import ResNet50
from keras_vggface.vggface import VGGFace


import os
import time

from PIL import Image
from keras import Input, Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, concatenate, Flatten
from skimage.feature import hog
from sklearn.metrics import confusion_matrix, roc_curve

import dnnlib
import pretrained_networks
import projector
from blink_detection import check_blink
from training import dataset, misc
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from skimage.transform import resize
def project_images_dataset(proj, dataset_name, data_dir, num_snapshots = 2):

    print('Loading images from "%s"...' % dataset_name)
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=1, repeat=False, shuffle_mb=0)

    all_ssim = []
    all_mse = []
    all_times = []
    labels = []
    image_idx = 0
    while (True):
        #print('Projecting image %d ...' % (image_idx), flush=True)
        try:
            images, label = dataset_obj.get_minibatch_np(1)
            labels.append(label)
        except:
            break
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        start = time.time()
        img,temp_ssim, temp_mse = project_image(proj, targets=images,img_num=image_idx, num_snapshots=num_snapshots)
        end = time.time()
        all_ssim.append(temp_ssim)
        all_mse.append(temp_mse)
        all_times.append(end-start)


        # print("Time to process image: ", end-start , flush=True)
        avg_time = sum(all_times)/len(all_times)
        image_idx += 1
        break
    return all_ssim, all_mse, labels,avg_time

def project_image(proj, targets, img_num, num_snapshots):
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    #misc.save_image_grid(targets, png_prefix + 'target.png', drange=[-1,1])
    proj.start(targets)
    while proj.get_cur_step() < proj.num_steps:
        print('\rProjecting image %d: %d / %d ... ' % (img_num,proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if proj.get_cur_step() == proj.num_steps:
            #misc.save_image_grid(proj.get_images(), png_prefix + 'step%04d.png' % proj.get_cur_step(), drange=[-1,1])

            imreal = np.array(misc.convert_to_pil_image(targets[0]))
            improj = np.array(misc.convert_to_pil_image(proj.get_images()[0]))
            imreal = normalize_img(imreal)
            improj = normalize_img(improj)
            temp_mse = mse(imreal,improj)
            temp_ssim = ssim(imreal,improj,multichannel=True)
            # print(temp_ssim, temp_mse , flush=True)

    # print(improj)
    # print(imreal)
    # misc.convert_to_pil_image(targets[0]).save("test_img_real.png")
    # misc.convert_to_pil_image(proj.get_images()[0]).save("test_img.png")
    return improj,temp_ssim, temp_mse
    print('\r%-30s\r' % '', end='', flush=True)


def normalize_img(pixels):
  #  pixels = pixels.astype('float32')
    mean, std = pixels.mean(), pixels.std()
    if np.isnan(std):
        return None
    pixels = (pixels - mean) / std
    pixels = (pixels - np.min(pixels)) / np.ptp(pixels)
    pixels = np.floor(pixels).astype('int16')
    return pixels

def test_network(weight,threshold_dataset_dir, test_dataset_dir,threshold_set,test_set, numsteps = 10 ,queue = None):
    print('Loading networks from "%s"...' %  weight)
    _G, _D, Gs = pretrained_networks.load_networks(weight)
    proj = projector.Projector()
    proj.set_network(Gs)
    proj.num_steps = numsteps

    #threshold_ssim, threshold_mse,threshold_labels,_ = project_images_dataset(proj,threshold_set,threshold_dataset_dir)

    # threshold_ssim_value = getThreshold(threshold_ssim,threshold_labels)
    # threshold_mse_value = getThreshold(threshold_mse,threshold_labels)


    test_ssim, test_mse, test_labels,avg_time = project_images_dataset(proj,test_set,test_dataset_dir)


    # thresholded_test_ssim = threshold_values(test_ssim,threshold_ssim_value,False)
    # thresholded_test_mse = threshold_values(test_mse,threshold_mse_value,True)
    # combined_measure_test = [x or y for x,y in zip(thresholded_test_mse, thresholded_test_ssim)]
    #
    # hter_ssim = calculate_metrics(thresholded_test_ssim,test_labels,"SSIM")
    # hter_mse = calculate_metrics(thresholded_test_mse,test_labels,"MSE")
    # hter_combined = calculate_metrics(combined_measure_test,test_labels,"Combined")
    hter_ssim = 0
    hter_mse = 0
    hter_combined = 0

    if queue is not None:
        queue.put((weight.split("-")[-1].split(".")[0], hter_ssim, hter_mse,hter_combined))
    return hter_ssim,hter_mse,hter_combined,avg_time


def getThreshold(predictions,labels):
    if max(labels) > 13:
        labels = [0 if x > 20 else 1 for x in labels]
    else:
        labels = [0 if x in [1,2,9] else 1 for x in labels]
    fpr, tpr, threshold = roc_curve(labels, predictions, pos_label=1)
    if sum(tpr)/len(tpr) < 0.5:
        fpr, tpr, threshold = roc_curve(labels, -np.array(predictions), pos_label=1)
        threshold = -np.array(threshold)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print(eer_threshold)
    print(eer)

    print("fpr = ", list(fpr))
    print("tpr = ", list(tpr))
    print("train_threshold = ", list(threshold))
    print("train_hter = " , [(fpr[i] + (1-tpr[i]))/2 for i in range(0,len(fpr))])

    return eer_threshold

#asc is true if the higher values are positive
def threshold_values(predictions,threshold, asc = True):
    if asc:
        return [0 if x < threshold else 1 for x in predictions]
    else:
        return [0 if x > threshold else 1 for x in predictions]


def calculate_metrics(thresholded_values,labels,name):
    print("Calculating metrics for method: " + name)

    true_labels = [x[0][0] for x in labels ]
    if max(true_labels) > 12:
        labels = [0 if x > 20 else 1 for x in labels]
    elif max(true_labels) < 2:
        labels = [x[0][0] for x in labels]
    elif max(true_labels) < 3:
        labels = [int(x) for x in labels]

    else:
        labels = [0 if x in [1,2,9] else 1 for x in labels]

    wrongly_classified = []
    for i in range(0,len(true_labels)):
        if thresholded_values[i] != labels[i]:
            wrongly_classified.append(true_labels[i])
    print("wrongly classified: ", wrongly_classified)




    conf = confusion_matrix(labels,thresholded_values)
    print(conf)
    TN = conf[0][0]
    FN = conf[1][0]
    TP = conf[1][1]
    FP = conf[0][1]

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



def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def test_different_nums():
    weight = "../weights/stylegan_training_weights/network-snapshot-015211.pkl"
    threshold_dataset_dir = "../databases/replay-attack/"
    test_dataset_dir = "../databases/casia-fasd/norm_scaled/"
    threshold_sets = "devel_faces_norm_scaled_tf"
    test_sets = "test_faces_norm_scaled_tf"

    nums = [1,2,5,10,20,50,100]
    all_ssim = []
    all_mse = []
    all_combined = []
    all_times = []
    for num in nums:
        temp_ssim,temp_mse,temp_combined,avg_time = test_network(weight, threshold_dataset_dir, test_dataset_dir, threshold_sets, test_sets,num)
        all_ssim.append(temp_ssim)
        all_mse.append(temp_mse)
        all_combined.append(temp_combined)
        all_times.append(avg_time)
    print("Results")
    print("nb It", "SSIM", "MSE", "Avg Time")
    for i in range(len(nums)):
        print(nums[i],all_ssim[i],all_mse[i],all_combined[i],all_times[i])

def test_diff_weights(weights):
    threshold_dataset_dir = "../databases/casia-fasd/faces/"
    test_dataset_dir = "../databases/replay-attack/"
    threshold_sets = "train_faces_tf"
    test_sets = "blink_dataset_replay"


    queue = Queue()
    all_results = []
    for weight in weights:

        p = multiprocessing.Process(target=test_full_network, args=(weight, threshold_dataset_dir, test_dataset_dir, threshold_sets, test_sets, 2,queue))
        p.start()
        all_results.append(queue.get())
        p.join()

    print("weight","SSIM","MSE","Combined_&","Combined_Scale_Sum","Neural_Network","Discriminator")
    for i in range(0,len(weights)):
        for j in range(0,len(all_results[i])):
            if j == 0:
                print(all_results[i][j], end= " ")
            elif j == len(all_results[i])-1:
                print(all_results[i][j] * 100)
            else:
                print(all_results[i][j] * 100, end= " ")



def gen_disc_test(proj, D, dataset_name, data_dir, num_snapshots = 2, queue = None):

    print('Loading images from "%s"...' % dataset_name)
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=2, repeat=False, shuffle_mb=0)

    all_ssim = []
    all_mse = []
    labels = []
    discrim = []
    image_idx = 0
    all_real_img = []
    all_proj_img = []

    all_blink_detects = []

    while (True):
        #print('Projecting image %d ...' % (image_idx), flush=True)
        # if image_idx == 10 :
        #     break
        try:
            images, label = dataset_obj.get_minibatch_np(1)
            # print(label)
            if not len(label[0]) == 1:
                label,name = label[0]
                label = np.array([[label]])
            else:
                name = None
            labels.append(label)
        except:
            break
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        img,temp_ssim, temp_mse = project_image(proj, targets=images,img_num=image_idx, num_snapshots=num_snapshots)
        all_ssim.append(temp_ssim)
        all_mse.append(temp_mse)
        all_real_img.append(np.array(misc.convert_to_pil_image(images[0])))
        all_proj_img.append(img)
        if name is not None:
            all_blink_detects.append(check_blink(parse_num(name,data_dir)))
        test = images
        # test = np.array(normalize_img(images))

        discrim.append(D.run(test, None)[0][0])

        image_idx += 1
    if queue is not None:
            queue.put((all_ssim, all_mse, labels,discrim,all_real_img,all_proj_img,all_blink_detects))
    return [all_ssim, all_mse, labels,discrim,all_real_img,all_proj_img,all_blink_detects]



def gen_disc_process(weight,numsteps, threshold_set, threshold_dataset_dir, test_set, test_dataset_dir, queue_proj = None):
    print('Loading networks from "%s"...' %  weight)
    _, D, Gs = pretrained_networks.load_networks(weight)
    proj = projector.Projector()
    proj.set_network(Gs)
    proj.num_steps = numsteps

    threshold_results = gen_disc_test(proj,D,threshold_set,threshold_dataset_dir)
    test_results = gen_disc_test(proj,D,test_set,test_dataset_dir)

    if queue_proj is not None:
        queue_proj.put((threshold_results,test_results))
    return threshold_results , test_results

def test_full_network(weight,threshold_dataset_dir, test_dataset_dir,threshold_set,test_set, numsteps = 10 ,queue = None):
    queue_proj = Queue()
    p = multiprocessing.Process(target=gen_disc_process, args=(weight,numsteps, threshold_set, threshold_dataset_dir, test_set, test_dataset_dir, queue_proj))
    p.start()
    threshold_res, test_res = queue_proj.get()
    p.join()

    threshold_ssim, threshold_mse, threshold_labels, discriminator_values, threshold_real_img, threshold_proj_img, detected_blinks = threshold_res
    test_ssim, test_mse, test_labels, test_discriminator_values, test_real_img, test_proj_img, test_detected_blinks = test_res


    normalized_mse = 1 - (threshold_mse - min(threshold_mse)) / max((threshold_mse - min(threshold_mse)))
    normalized_disc_values = (discriminator_values - min(discriminator_values)) / max((discriminator_values - min(discriminator_values)))
    combined_sum = [x + y for x, y in zip(threshold_ssim, normalized_mse)]
    normalized_combined_sum =  (combined_sum - min(combined_sum)) / max((combined_sum - min(combined_sum)))
    full_network_combined_sum = [x + y for x, y in zip(normalized_combined_sum, normalized_disc_values)]

    print("threshold ssim: ")
    threshold_ssim_value = getThreshold(threshold_ssim,threshold_labels)
    print("threshold mse: ")
    threshold_mse_value = getThreshold(threshold_mse,threshold_labels)
    print("threshold discriminator: ")
    threshold_discriminator = getThreshold(discriminator_values,threshold_labels)
    print("threshold combined sum")
    threshold_combined = getThreshold(combined_sum,threshold_labels)
    print("threshold full network")
    threshold_full_network_combined = getThreshold(full_network_combined_sum,threshold_labels)


    if max(threshold_labels) > 12:
        labels = [0 if x > 20 else 1 for x in threshold_labels]
    else:
        labels = [0 if x in [1,2,9] else 1 for x in threshold_labels]

    if max(test_labels) > 12:
        val_labels = [0 if x > 20 else 1 for x in test_labels]
    else:
        val_labels = [0 if x in [1,2,9] else 1 for x in test_labels]


    rescaled_th_real = [resize(x, (224, 224, 3)) for x in threshold_real_img]
    rescaled_th_proj = [resize(x, (224, 224, 3)) for x in threshold_proj_img]
    rescaled_test_real = [resize(x, (224, 224, 3)) for x in test_real_img]
    rescaled_test_proj = [resize(x, (224, 224, 3)) for x in test_proj_img]


    model = train_model(rescaled_th_real,rescaled_th_proj,rescaled_test_real,rescaled_test_proj,labels,val_labels)


    normalized_mse = 1 - (test_mse - min(test_mse)) / max((test_mse - min(test_mse)))
    normalized_disc_values_test = (test_discriminator_values - min(test_discriminator_values)) / max((test_discriminator_values - min(test_discriminator_values)))
    combined_sum = [x + y for x, y in zip(test_ssim, normalized_mse)]
    normalized_combined_sum =  (combined_sum - min(combined_sum)) / max((combined_sum - min(combined_sum)))
    full_network_combined_sum_test = [x + y for x, y in zip(normalized_combined_sum, normalized_disc_values_test)]


    thresholded_test_ssim = threshold_values(test_ssim,threshold_ssim_value,False)
    thresholded_test_mse = threshold_values(test_mse,threshold_mse_value,True)
    thresholded_test_discriminator = threshold_values(test_discriminator_values,threshold_discriminator,False)
    thresholded_combined_sum = threshold_values(combined_sum,threshold_combined,False)
    thresholded_full_network_combined = threshold_values(full_network_combined_sum_test,threshold_full_network_combined,False)


    print("threshold siamese network")
    siamese_threshold_values = [model.predict([[x],[y]])[0][0] for x,y in zip(rescaled_th_real,rescaled_th_proj) ]
    siamese_threshold = getThreshold(siamese_threshold_values,threshold_labels)
    siamese_test_values = [model.predict([[x],[y]])[0][0] for x,y in zip(rescaled_test_real,rescaled_test_proj) ]
    siamese_test = threshold_values(siamese_test_values,siamese_threshold,False)

    combined_network_test = [x and y for x,y in zip(thresholded_combined_sum, thresholded_test_discriminator)]

    print("ROC values combined sum")
    print_roc_values(combined_sum,val_labels)
    print("ROC values MSE")
    print_roc_values(test_mse,val_labels)
    print("ROC values SSIM")
    print_roc_values(test_ssim,val_labels)
    print("ROC values Full network combined")
    print_roc_values(full_network_combined_sum_test,val_labels)
    print("ROC values full network &&")
    print_roc_values(combined_network_test,val_labels)
    print("ROC values siamese network")
    print_roc_values(siamese_test_values,val_labels)


    added_blink_test = [(not x) or y for x,y in zip(test_detected_blinks,combined_network_test)]
    added_blink_test_with_scaled_sum = [(not x) or y for x,y in zip(test_detected_blinks,thresholded_full_network_combined)]



    hter_ssim = calculate_metrics(thresholded_test_ssim,test_labels,"SSIM")
    hter_mse = calculate_metrics(thresholded_test_mse,test_labels,"MSE")


    hter_combined = calculate_metrics(siamese_test,test_labels,"Siamese_nn")
    hter_combined_sum = calculate_metrics(thresholded_combined_sum,test_labels,"combined_sum")
    hter_discriminator = calculate_metrics(thresholded_test_discriminator,test_labels,"discriminator")
    hter_network = calculate_metrics(combined_network_test,test_labels,"network")
    hter_combined_network= calculate_metrics(thresholded_full_network_combined,test_labels,"scaled_sum_network")
    hter_just_blinks = calculate_metrics([not x for x in test_detected_blinks],test_labels,"Just blinky")
    hter_blink_test = calculate_metrics(added_blink_test,test_labels,"Blinky Test")
    hter_blink_test_scaled_sum = calculate_metrics(added_blink_test_with_scaled_sum,test_labels,"Blinky Test with scaled sum GAN")

    if queue is not None:
        queue.put((weight.split("-")[-1].split(".")[0], hter_ssim, hter_mse,hter_combined,hter_combined_sum,hter_network,hter_discriminator))
    return hter_ssim,hter_mse,hter_combined,hter_network


def train_model(rescaled_th_real,rescaled_th_proj,rescaled_test_real,rescaled_test_proj,labels,val_labels,queue=None):
    print("started model")
    # base_network = create_base_network((128,128,3))

    # real_img_input = Input(shape=(128,128,3))
    # proj_img_input = Input(shape=(128,128,3))

    real_img_input = Input(shape=(224, 224, 3))
    proj_img_input = Input(shape=(224, 224, 3))

    # processed_real = base_network(real_img_input)
    # processed_proj = base_network(proj_img_input)
    processed_real = VGGFace(include_top=False, input_tensor=real_img_input)
    processed_proj = VGGFace(include_top=False, input_tensor=proj_img_input)
    for layer in processed_proj.layers:
        layer.name = layer.name + str("proj")
        layer.trainable = False
    for layer in processed_real.layers:
        layer.trainable = False


    merged = concatenate([processed_real.output, processed_proj.output], axis=-1)

    x = Flatten()(merged)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    prediction = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[real_img_input, proj_img_input], outputs=prediction)

    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit([np.stack(threshold_real_img),np.stack(threshold_proj_img)], labels, epochs=3, validation_data=([np.stack(test_real_img),np.stack(test_proj_img)],np.stack(val_labels)))
    model.fit([np.stack(rescaled_th_real), np.stack(rescaled_th_proj)], labels, epochs=10,
              validation_data=([np.stack(rescaled_test_real), np.stack(rescaled_test_proj)], np.stack(val_labels)))
    if queue is not None:
        queue.put(model)
    return model

def create_base_network(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(64,(5, 5), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    return Model(input_img, x)


def print_roc_values(predictions,labels):

    predictions = [abs(x) for x in predictions]
    print("pred = " , list(zip(predictions,labels)))
    fpr, tpr, threshold = roc_curve(labels, predictions, pos_label=1)
    if sum(tpr)/len(tpr) < 0.5:
        fpr, tpr, threshold = roc_curve(labels, -np.array(predictions), pos_label=1)
        threshold = -np.array(threshold)


    print("fpr = ", list(fpr))
    print("tpr = ", list(tpr))
    print("threshold = ", list(threshold))
    print("hter = " , [(fpr[i] + (1-tpr[i]))/2 for i in range(0,len(fpr))])

def apply_hog(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)

    # hog_image = [[[x, x, x] for x in line] for line in hog_image]
    # image = np.multiply(image, hog_image)
    # image *= 255.0 / image.max()
    # image = np.floor(image)
    return [hog_image]

def parse_num(num, dataset):
    person = str(int(num))[:-2]
    vid_num = int(str(int(num))[-2:])
    if "casia" in dataset:
        vid_name = str(vid_num)  if vid_num < 9 else "HR_" + str(vid_num - 8)
        path_to_vids = "../databases/casia-fasd/test/test_release"
        return path_to_vids + "/" + person + "/" + vid_name + ".avi"
    if "replay" in dataset:
        path_to_vids = "../databases/replay-attack/test"
        attack_or_real = "attack" if vid_num < 21 else "real"
        adverse_controlled = "controlled" if vid_num in [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22] else "adverse"
        if attack_or_real == "attack":

            if vid_num in   [1, 6, 11, 16]:
                full_name = "attack_highdef_client" + str(person.zfill(3)) + "_session01_" + "highdef_photo_" + adverse_controlled + ".mov"
            elif vid_num in [2, 7, 12, 17]:
                full_name = "attack_highdef_client" + str(person.zfill(3)) + "_session01_" + "highdef_video_" + adverse_controlled + ".mov"
            elif vid_num in [3, 8, 13, 18]:
                full_name = "attack_mobile_client" + str(person.zfill(3)) + "_session01_" + "mobile_photo_" + adverse_controlled + ".mov"
            elif vid_num in [4, 9, 14, 19]:
                full_name = "attack_mobile_client" + str(person.zfill(3)) + "_session01_" + "mobile_video_" + adverse_controlled + ".mov"
            elif vid_num in [5, 10, 15, 20]:
                full_name = "attack_print_client" + str(person.zfill(3)) + "_session01_" + "highdef_photo_" + adverse_controlled + ".mov"

            attack_mode = "fixed" if vid_num < 11 else "hand"
            return path_to_vids + "/" + attack_or_real + "/" + attack_mode + "/" + full_name

        elif attack_or_real == "real":
            full_name = "client" + str(person.zfill(3)) + "_session01_webcam_authenticate_" + adverse_controlled + "_" + str(vid_num - 20 if adverse_controlled == "controlled" else vid_num - 22) + ".mov"
            return path_to_vids + "/" + attack_or_real + "/" + full_name



path = "../weights/stylegan2_straight_faces/"

weights = os.listdir(path)
weights = [os.path.join(path,x) for x in weights if x.endswith(".pkl")]
weights = [sorted(weights)[-1]]
test_diff_weights(weights)

#test_different_nums()
# dnnlib.tflib.init_tf()

