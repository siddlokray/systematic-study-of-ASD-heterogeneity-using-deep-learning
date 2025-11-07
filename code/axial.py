# import libraries
import tensorflow as tf # tensorflow
from tensorflow.keras.models import Sequential, Model # to build model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, ZeroPadding2D # layers
import nibabel as nib # nifti
import os # access directories
import numpy as np # array operations
import matplotlib.pyplot as plt # data visualisation
from sklearn.preprocessing import LabelEncoder # encoding
from keras.utils import np_utils # also encoding
from sklearn.model_selection import train_test_split # split data
from scipy.ndimage import zoom # resize 3d data
import pandas as pd # csv
import math # math
from scipy.ndimage import gaussian_filter # smoothing
import cv2 as cv

class var:
    lr = 3e-3
    volume = 'nu-to-mni305.mgz'
    notes = 'smoothing, 3, 1, random state 0, reduced data, men, 12 y/o, asd v non asd'
    imsize = 128
    factor = imsize/256
    fwhm = 3
    voxsize = 1
    z = 64
    

# print stuff
print('LEARNING RATE: '+str(var.lr))
print('VOLUME: '+str(var.volume))
print('NOTES: '+str(var.notes))
print('IMAGE SIZE: ' + str(var.imsize))

# get filenames and labels into lists
# uncomment lines to read csv
flist = [
         '/projectnb/nickar/freesurfer/csv/ABIDEII-NYU_1.csv',
         '/projectnb/nickar/freesurfer/csv/ABIDEII-SDSU_1.csv',
         '/projectnb/nickar/freesurfer/csv/ABIDEII-TCD_1.csv',
#          '../input/abideii-subset/scan data/ABIDEII-ETH_1.csv',
         '/projectnb/nickar/freesurfer/csv/ABIDEII-NYU_2.csv',
         '/projectnb/nickar/freesurfer/csv/ABIDEII-OHSU_1.csv',
         '/projectnb/nickar/freesurfer/csv/ABIDEII-KKI_1.csv'
#          '../input/abideii-subset/scan data/ABIDEII-BNI_1.csv',
#          '../input/abideii-subset/scan data/ABIDEII-UCD_1.csv',
#          '../input/abideii-subset/scan data/ABIDEII-USM_1.csv'
        ]
f = [] # filenames
l = [] # labels
# iterate through csv adding filenames and labels to respective lists
# checks if the filename exists and labels the filename based on SRS_TOTAL_T score
# using intensity normalization and spatial normalization mris
for college_csv in flist:
    csv = pd.read_csv(college_csv)
    csv = csv[['SUB_ID', 'DX_GROUP', 'AGE_AT_SCAN ', 'SEX', 'SRS_EDITION','SRS_TOTAL_T']]
    s = college_csv.split('/')[5]
    s = s.split('.')[0]
    for i, r in csv.iterrows():
        p = os.path.join('/projectnb/nickar/freesurfer', str(int(r[0])), str(var.volume))
        if os.path.exists(p):
            try:
                if math.isnan(r[5]):
                    pass
                else:
                    if (r[2] <= 12) and (r[3] == 1):
                        if r[5] < 45:
                            f.append(p)
                            l.append(0)
                        elif r[5] > 70:
                            f.append(p)
                            l.append(1)
                        # else:
                        #     f.append(p)
                        #     l.append(2)
            except Exception as e:
                print(r[0])
        else:
            pass


ll = l




# encode
encoder = LabelEncoder()
l = encoder.fit_transform(l)
l = np_utils.to_categorical(l)

# split data into train and test
xtrain, xtest, ytrain, ytest = train_test_split(f, l, train_size=0.80, random_state=0, stratify=l)

train_big_f = []
train_big_l = []
for i in range(len(xtrain)):
    z = var.z
    slices = [z,z+1,z+2,z+3,z+4,z+5,z+6,z+7]
    filename = xtrain[i]
    label = ytrain[i]
    for j in slices:
        slice_file = filename + ' ' + str(j)
        train_big_f.append(slice_file)
        train_big_l.append(label)
test_big_f = []
test_big_l = []
for i in range(len(xtest)):
    z = var.z
    slices = [z,z+1,z+2,z+3,z+4,z+5,z+6,z+7]
    filename = xtest[i]
    label = ytest[i]
    for j in slices:
        slice_file = filename + ' ' + str(j)
        test_big_f.append(slice_file)
        test_big_l.append(label)

# preprocessing and dataset
# allows for using array options instead of tensor operations
def tf_parse(filename, label):
    path = filename
    [image,] = tf.py_function(p, [path], [tf.uint8])
    return image, label

# preprocessing here
# smoothing
def smooth(v, fwhm, vs):
    sigma = fwhm / (np.sqrt(8 * np.log(2)) * vs)
    sm = gaussian_filter(v, sigma=sigma)
    return sm

# load mri from filepath
# pad mri to anatomical space
# resize mri to smaller size
def p(path):
    path = bytes.decode(path.numpy())
    path, volslice = path.split(' ')
    image = nib.load(path)
    image = image.get_fdata()
    image = image.T
    image = np.transpose(image, (0,2,1))
    image = np.flip(image, axis=2)
    x, y, z = image.shape
    image = np.pad(image, ((0,256-x),(0,256-y),(0,256-z)), mode='constant', constant_values=0)
    image = smooth(image, var.fwhm, var.voxsize)
    image = zoom(image, (var.factor,var.factor,var.factor))
    image = image[:,:,int(volslice)]
    image = np.reshape(image, (128,128,1))
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.uint8)
    return image

# load train dataset
dataset = tf.data.Dataset.from_tensor_slices((train_big_f, train_big_l))
dataset = dataset.shuffle(len(train_big_f))
dataset = dataset.map(tf_parse, num_parallel_calls=1)
dataset = dataset.batch(1)
dataset = dataset.prefetch(1)

# load validation dataset
tdataset = tf.data.Dataset.from_tensor_slices((test_big_f, test_big_l))
tdataset = tdataset.shuffle(len(test_big_f))
tdataset = tdataset.map(tf_parse, num_parallel_calls=1)
tdataset = tdataset.batch(1)
tdataset = tdataset.prefetch(1)

# resnet 3d architecture
from tensorflow import keras
from tensorflow.keras import layers

# resnet identity block
def res_identity(x, filters): 
    x_skip = x # this will be used for addition with the residual block 
    f1, f2 = filters
    x = layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    #second block # bottleneck (but size kept same with padding)
    x = layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # third block activation used after adding the input
    x = layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    # x = Activation(activations.relu)(x)
    # add the input 
    x = layers.Add()([x, x_skip])
    x = layers.Activation('relu')(x)
    return x

# resnet convolution block
def res_conv(x, s, filters):
    x_skip = x
    f1, f2 = filters
    # first block
    x = layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid')(x)
    # when s = 2 then it is like downsizing the feature map
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # second block
    x = layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    #third block
    x = layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    # shortcut 
    x_skip = layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid')(x_skip)
    x_skip = layers.BatchNormalization()(x_skip)
    # add 
    x = layers.Add()([x, x_skip])
    x = layers.Activation('relu')(x)
    return x

# resnet50
def resnet50():
    inputs = keras.Input((var.imsize,var.imsize,1))
    # 1st stage
    # here we perform maxpooling, see the figure above
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    #2nd stage 
    # frm here on only conv block and identity block, no pooling
    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    # 3rd stage
    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    # 4th stage
    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    # 5th stage
    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    # ends with average pooling and dense connection
    x = layers.AveragePooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    # add fc layers
#    x = layers.Dense(2, activation='relu')(x)
#     x = layers.Dense(512, activation='relu')(x)
#     x = layers.Dropout(0.3)(x)
    output = layers.Dense(2, activation='softmax', kernel_initializer='he_normal')(x) #multi-class
#     # define the model 
    model = keras.Model(inputs=inputs, outputs=output, name='Resnet50-2D')
    return model

# load model
model = resnet50()

# compile model
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=var.lr), metrics=['accuracy'])

# early stopping callback
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

# train model
model.fit(dataset, validation_data=tdataset, epochs=100, verbose=1
          , callbacks=[es_callback]
         )

model.evaluate(tdataset)

