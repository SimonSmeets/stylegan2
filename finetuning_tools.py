import os
from PIL import Image

from test_discriminator import get_all_images
import numpy as np
from skimage.transform import resize
from dataset_tool import create_from_images
import matplotlib
import shutil
def dataset_to_res(loc, output_loc,res):
    print(loc)
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    for path in os.listdir(loc):
        if os.path.isdir(os.path.join(loc,path)):
            try:
                os.mkdir(os.path.join(output_loc,path))
            except OSError:
                pass
            dataset_to_res(os.path.join(loc,path),os.path.join(output_loc,path),res)
        else:
            dir_to_frames(loc,output_loc,res)
            break
            print("dir done")


def dir_to_frames(input_loc, output_loc, res):
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    for imagename in os.listdir(input_loc):
            im = Image.open(os.path.join(input_loc,imagename))
            image = np.array(im)
            image = resize(image, (res,res))
            matplotlib.pyplot.imsave(os.path.join(output_loc,imagename),image)


def split_casia_finetuning(loc, output_loc):
    print(loc)
    try:
        os.mkdir(os.path.join(os.path.join(output_loc,"real/")))
        os.mkdir(os.path.join(os.path.join(output_loc,"attack/")))
    except OSError:
        pass
    for path in os.listdir(loc):
        if os.path.isdir(os.path.join(loc,path)):
            try:
                os.mkdir(os.path.join(os.path.join(output_loc,"real/"),os.path.split(path)[-1]))
                os.mkdir(os.path.join(os.path.join(output_loc,"attack/"),os.path.split(path)[-1]))

            except OSError:
                pass
            split_casia_finetuning(os.path.join(loc,path),output_loc)
        else:
            if path.startswith(("1","2","HR_1")):
                shutil.copy2(os.path.join(loc,path),os.path.join(os.path.join(output_loc,"real/"),os.path.join(os.path.split(loc)[-1],path)))
            else:
                shutil.copy2(os.path.join(loc,path),os.path.join(os.path.join(output_loc,"attack/"),os.path.join(os.path.split(loc)[-1],path)))
    print("dir done")


res = 128
path = "../databases/casia-fasd/train_frames"
#dataset_to_res(path,"../databases/casia-fasd/train_frames_128",128)
#split_casia_finetuning("../databases/casia-fasd/train_frames_128/train_release", "../databases/casia-fasd/train_frames_128")
create_from_images("../databases/casia-fasd/real_tfrecords","../databases/casia-fasd/train_frames_128/real/*",True)

#test_images, test_labels = get_all_images(path, "attack")
