import keras
import keras.preprocessing.image as image
import numpy as np
from PIL import Image
from keras import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
from skimage.metrics import structural_similarity as ssim

from test_discriminator import get_all_images
from test_generator import getThreshold, threshold_values, calculate_metrics


#tested before generator was used


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = image.img_to_array(image.load_img('../databases/thumbnails128x128/' + str(ID).zfill(5)[:2].zfill(2) + "000/" + str(ID).zfill(5) + '.png'))/255
            #y[i] = self.labels[ID]

        return X, X

def train_autoencoder():
    params = {'dim': (128,128),
              'batch_size': 64,
              'n_classes': 1,
              'n_channels': 3,
              'shuffle': True}

    partition = [x for x in range(0,60000)]
    label = [1 for x in range(0,60000)]

    partition2 = [x for x in range(60000,70000)]
    label2 = [1 for x in range(60000,70000)]

    training_generator = DataGenerator(partition, label, **params)
    test_generator = DataGenerator(partition2,label2, **params)


    input_img = Input(shape=(128,128,3))  # adapt this if using `channels_first` image data format

    x = Conv2D(128, (5, 5), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 64) i.e. 1024-dimensional
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.summary()
    autoencoder = multi_gpu_model(autoencoder, gpus=4)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit_generator(generator=training_generator,validation_data=test_generator, epochs=100)

    autoencoder.save("../autoencoder_results/test_autoencoder_larger_network.h5")



def test_autoencoder():
    autoencoder = load_model("test_autoencoder_larger_network.h5")

    threshold_set, threshold_labels = get_all_images(r'../databases/casia-fasd/faces/train_faces', "attack")
    test_set, test_labels = get_all_images(r'../databases/replay-attack/faces/test_faces', "attack")


    threshold_mse = []
    threshold_ssim = []
    for img in threshold_set:
        real_img = image.img_to_array(image.load_img(img))
        projection = get_projection(autoencoder,real_img)
        threshold_mse.append(mse(real_img, projection))
        threshold_ssim.append(ssim(real_img, projection, multichannel=True))

    test_mse = []
    test_ssim = []
    for img in test_set:
        real_img = image.img_to_array(image.load_img(img))
        projection = get_projection(autoencoder,real_img)
        test_mse.append(mse(real_img, projection))
        test_ssim.append(ssim(real_img, projection, multichannel=True))

    threshold_labels = [np.array([np.array([x])]) for x in threshold_labels]
    threshold_ssim_value = getThreshold(threshold_ssim,threshold_labels)
    threshold_mse_value = getThreshold(threshold_mse,threshold_labels)

    thresholded_test_ssim = threshold_values(test_ssim,threshold_ssim_value,False)
    thresholded_test_mse = threshold_values(test_mse,threshold_mse_value,True)
    combined_measure_test = [x or y for x,y in zip(thresholded_test_mse, thresholded_test_ssim)]

    hter_ssim = calculate_metrics(thresholded_test_ssim,test_labels,"SSIM")
    hter_mse = calculate_metrics(thresholded_test_mse,test_labels,"MSE")
    hter_combined = calculate_metrics(combined_measure_test,test_labels,"Combined")

    print(hter_ssim,hter_mse,hter_combined)



def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def get_projection(autoencoder,real_img):
    real_img.resize(1, 128, 128, 3)
    normalized_img = real_img/255
    projection = autoencoder.predict(normalized_img)
    real_img.resize(128,128,3)
    return projection[0]*255
#
# autoencoder = load_model("test_autoencoder.h5")
# X = np.empty((1, 128,128, 3))
# X[0,] = image.img_to_array(image.load_img('../databases/replay-attack/devel_faces/attack/fixed/attack_highdef_client003_session01_highdef_photo_adverse.png'))/255
# test_output = autoencoder.predict(X)[0]*255



# Image.fromarray(test_output.astype(np.uint8)).save("test_image_autoencoder.png")

# train_autoencoder()


test_autoencoder()
