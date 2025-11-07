import tensorflow as tf # tensorflow
from tensorflow.keras.models import Sequential, Model # to build model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv3D, Dense, Dropout, Flatten, MaxPool3D, ZeroPadding3D # layers
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
import keras

def average(subj,file):
    try:
        #     nu_path = os.path.join('/projectnb/nickar/corticalflatmap/sub1/freesurfer/segmentations',
        #                               str(subj),
        #                               'nu-to-mni305.mgz')
            nu = nib.load(mri_filename)
            nu = nu.get_fdata()
            nu = nu.T
            nu = np.transpose(nu,(0,1,2))
            nu = np.flip(nu, axis=1)
            # nu = zoom(nu,(0.5,0.5,0.5))
        #     vol_path = os.path.join('/projectnb/nickar/corticalflatmap/sub1/freesurfer/segmentations',
        #                               str(subj),
        #                               'volume.nii')
        #     vol = nib.load(vol_path)
        #     vol = vol.get_fdata()
        #     vol = zoom(vol,(2,2,2))
        #     sal_path = os.path.join('/projectnb/nickar/corticalflatmap/sub1/freesurfer/segmentations',
        #                               str(subj),
        #                               'important.nii')
        #     sal = nib.load(sal_path)
        #     sal = sal.get_fdata()
            sal = zoom(saliency,(2,2,2))
            mask_path = os.path.join('/projectnb/nickar/freesurfer',
                                      str(int(subj)),
                                      file)
            mask = nib.load(mask_path)
            mask = mask.get_fdata()
            mask = mask.T
            mask = np.transpose(mask,(0,1,2))
            mask = np.flip(mask, axis=1)

            masked = np.zeros_like(mask)
            val = []
            total = [1e-18]
            for x in range(nu.shape[0]):
                for y in range(nu.shape[1]):
                    for z in range(nu.shape[2]):
                        if mask[x,y,z] == 1.0:
                            masked[x,y,z] = sal[x,y,z]
                            val.append(sal[x,y,z])
                            total.append(1)
            print('subject: ' + str(subj) + ' mask: ' + str(file))
            av = sum(val)/sum(total)
            print('average saliency: ' + str(av))
            persubject.append(av)
    except Exception as e:
        print(e)
        persubject.append('nan')

def load_weights():
    def res_identity(x, filters): 
        x_skip = x # this will be used for addition with the residual block 
        f1, f2 = filters
        x = layers.Conv3D(f1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        #second block # bottleneck (but size kept same with padding)
        x = layers.Conv3D(f1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # third block activation used after adding the input
        x = layers.Conv3D(f2, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
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
        x = layers.Conv3D(f1, kernel_size=(1, 1, 1), strides=(s, s, s), padding='valid')(x)
        # when s = 2 then it is like downsizing the feature map
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # second block
        x = layers.Conv3D(f1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        #third block
        x = layers.Conv3D(f2, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')(x)
        x = layers.BatchNormalization()(x)
        # shortcut 
        x_skip = layers.Conv3D(f2, kernel_size=(1, 1, 1), strides=(s, s, s), padding='valid')(x_skip)
        x_skip = layers.BatchNormalization()(x_skip)
        # add 
        x = layers.Add()([x, x_skip])
        x = layers.Activation('relu')(x)
        return x

    # resnet50
    def resnet50():
        inputs = keras.Input((128,128,128,1))
        # 1st stage
        # here we perform maxpooling, see the figure above
        x = layers.Conv3D(64, kernel_size=(7, 7, 7), strides=(2, 2, 2))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(x)
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
        x = layers.AveragePooling3D((2, 2, 2), padding='same')(x)
        x = layers.Flatten()(x)
        # add fc layers
    #    x = layers.Dense(2, activation='relu')(x)
    #     x = layers.Dense(512, activation='relu')(x)
    #     x = layers.Dropout(0.3)(x)
        output = layers.Dense(2, activation='softmax', kernel_initializer='he_normal')(x) #multi-class
    #     # define the model 
        model = keras.Model(inputs=inputs, outputs=output, name='Resnet50-3D')
        return model

    # load model
    model = resnet50()
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=3e-3), metrics=['accuracy'])
    model.load_weights('/projectnb/nickar/sidd/code/training-1/cp.ckpt').expect_partial()
    return model
def smooth(v, fwhm, vs):
    sigma = fwhm / (np.sqrt(8 * np.log(2)) * vs)
    sm = gaussian_filter(v, sigma=sigma)
    return sm
def preprocess(path):
    image = nib.load(path)
    image = image.get_fdata()
    image = image.T
    image = np.transpose(image, (0,2,1))
    image = np.flip(image, axis=2)
    x, y, z = image.shape
    image = np.pad(image, ((0,256-x),(0,256-y),(0,256-z)), mode='constant', constant_values=0)
    image = smooth(image, 3, 2)
    image = zoom(image, (0.5,0.5,0.5))
    image = np.reshape(image, (1,128,128,128,1))
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    return image
def get_saliency_map(image,q,w,e):
    image = tf.Variable(image, dtype=float)
    with tf.GradientTape() as tape:
        pred = model(image, training=False)
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]
    grads = tape.gradient(loss, image)
    dgrad_abs = tf.math.abs(grads)
    dgrad_max_ = np.max(dgrad_abs, axis=4)[0]
    arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
    # grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
    # grad_eval = np.transpose(dgrad_max_, (q,w,e))
    return dgrad_max_
def average_volume(sal):
    sal_min, sal_max  = np.min(sal), np.max(sal)
    sal = (sal - sal_min) / (sal_max - sal_min + 1e-18)
    sal = (sal*255).astype('uint8')
#     for i in range(128):
#         if i == 0 or i == 1 or i == 126 or i == 127:
#             pass
#         else:
#             sal[:,:,i] = average(i,sal)
    return sal
def extract_saliency(filename):
    volume = preprocess(filename)
    saliency = get_saliency_map(volume,0,1,2)
    normalized_saliency = average_volume(saliency)
    normalized_saliency = np.flip(normalized_saliency, axis=2)
    normalized_saliency = np.transpose(normalized_saliency, (0,2,1))
    saliency = normalized_saliency.T
    return saliency

flist = [
         '/projectnb/nickar/freesurfer/csv/ABIDEII-NYU_1.csv',
         '/projectnb/nickar/freesurfer/csv/ABIDEII-SDSU_1.csv',
         '/projectnb/nickar/freesurfer/csv/ABIDEII-TCD_1.csv',
         '/projectnb/nickar/freesurfer/csv/ABIDEII-NYU_2.csv',
         '/projectnb/nickar/freesurfer/csv/ABIDEII-OHSU_1.csv',
         '/projectnb/nickar/freesurfer/csv/ABIDEII-KKI_1.csv'
        ]
f = [] # filenames
l = [] # labels
subj = []
#files = [
#        6,16,19,20,26,27,45,55,56,58,59,250,251,252,253,254,255,
#        1004,1010,1013,1014,1015,1016,1018,1019,1020,1022,1023,1025,1026,1028,
#        1029,1030,1031,1032,1033,1034,1035,
#        2004,2010,2013,2014,2015,2016,2018,2019,2020,2022,2023,2025,2026,2028,2029,2030,2031,2032,2033,2034,2035,
#        2,7,8,10,11,12,13,17,18,41,46,47,49,50,51,52,53,54,
#        1002,1003,1005,1006,1007,1008,1009,1011,1012,1017,1021,1024,1027,
#        2002,2003,2005,2006,2007,2008,2009,2011,2012,2017,2021,2024,2027
#        ]
files = [1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,
    1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1137,1138,1139,
    1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,
    1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,
    1180,1181,1200,1201,1202,1205,1206,1207,1210,1211,1212,2101,2102,2103,2104,2105,2106,2107,2108,2109,
    2110,2111,2112,2113,2114,2115,2116,2117,2118,2119,2120,2121,2122,2123,2124,2125,2126,2127,2128,2129,
    2130,2131,2132,2133,2134,2135,2136,2137,2138,2139,2140,2141,2142,2143,2144,2145,2146,2147,2148,2149,
    2150,2151,2152,2153,2154,2155,2156,2157,2158,2159,2160,2161,2162,2163,2164,2165,2166,2167,2168,2169,
    2170,2171,2172,2173,2174,2175,2176,2177,2178,2179,2180,2181,2200,2201,2202,2205,2206,2207,2210,2211,
    2212
    ]
files = sorted(files)

for college_csv in flist:
    csv = pd.read_csv(college_csv)
    csv = csv[['SUB_ID', 'DX_GROUP', 'AGE_AT_SCAN ', 'SEX', 'SRS_EDITION','SRS_TOTAL_T']]
    s = college_csv.split('/')[5]
    s = s.split('.')[0]
    for i, r in csv.iterrows():
        p = os.path.join('/projectnb/nickar/freesurfer', str(int(r[0])), str('nu-to-mni305.mgz'))
        p1 = os.path.join('/projectnb/nickar/freesurfer', str(int(r[0])), str('mri/aparc.DKTatlas+aseg.mgz'))
        if os.path.exists(p):
            if os.path.exists(p1):
                try:
                    if math.isnan(r[5]):
                        pass
                    else:
                        if (r[2] <= 12) and (r[3] == 1):
                            if r[5] < 45:
                                f.append(p)
                                l.append(0)
                                subj.append(r[0])
                            elif r[5] > 70:
                                f.append(p)
                                l.append(1)
                                subj.append(r[0])
                            # else:
                            #     f.append(p)
                            #     l.append(2)
                except Exception as e:
                    print(r[0])
        else:
            pass

model = load_weights()

print('loaded weights')

df = pd.DataFrame(columns=subj)
df['AREA'] = files
for sub in subj:
    mri_filename = '/projectnb/nickar/freesurfer/' + str(int(sub)) + '/nu-to-mni305.mgz'
    saliency = extract_saliency(mri_filename)
    persubject = []
    for file in files:
        file = str(file) + '.mgz'
        average(sub,file)
    df[sub] = persubject

df.to_csv('/projectnb/nickar/sidd/fixed_pipeline_destrieux.csv')
