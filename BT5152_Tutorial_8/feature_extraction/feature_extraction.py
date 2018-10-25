"""
Using Keras’ Pre-trained Models for Feature Extraction
"""

from keras.models import Model
from keras.preprocessing import image

import numpy as np
import pandas as pd
import csv
import os
import time

current_dir = os.getcwd()


######################################
### feature extraction from ResNet ###
######################################
"""
Default input size:
Layer_name: specify the layer from which features are extracted
Output: features extracted (flattened) from the specified layer
"""
# Load ResNet 50
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

base_model = ResNet50(weights='imagenet', include_top=False)
base_model.summary()

## not specify layer, by default use the last layer before softmax
model = base_model

img_h, img_w = 224, 224
# iterate through plots and extract features
start = time.time()
with open(current_dir + '/feature_extraction/features/resnet50_feature.csv', 'w') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    for file_name in os.listdir('feature_extraction/plots/'):
        print(file_name)
        img = image.load_img(current_dir + '/feature_extraction/plots/'+file_name, target_size=(img_h, img_w))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        feature = model.predict(img_data)
        # flatten the output feature and write it to csv file
        feature_np = np.array(feature)
        row = list(feature_np.flatten())
        row.append(file_name)
        writer.writerow(row)
end = time.time()
print('Time taken:', end - start)

features_df = pd.read_csv('feature_extraction/features/resnet50_feature.csv')
features_df.shape

###########################################
### feature extraction from VGG16/VGG19 ###
###########################################
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
"""
Using VGG16/VGG19 Pre-trained Models for Feature Extraction
Default input size: 224*224 RGB image
Layer_name: specify the layer from which features are extracted
Output: features extracted (flattened) from the specified layer
"""
# load VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)
base_model.summary()
# specify the VGG16 layer from which features are extracted
# you can choose not to use the 3 lines below, then by default it will extract the features of
# the last dense layer before softmax classification layer
layer_name = 'block5_conv3'
model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
model.summary()
# specify input image size
img_h, img_w = 224, 224
# iterate through plots and extract features
start = time.time()
with open(current_dir + '/feature_extraction/features/vgg16_feature.csv', 'w') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    for file_name in os.listdir(current_dir +'/feature_extraction/plots/'):
        print(file_name)
        img = image.load_img(current_dir + '/feature_extraction/plots/'+file_name, target_size=(img_h, img_w))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        feature = model.predict(img_data)
        # flatten the output feature and write it to csv file
        feature_np = np.array(feature)
        row = list(feature_np.flatten())
        row.append(file_name)
        writer.writerow(row)
end = time.time()
print('Time taken:', end - start)

features_df = pd.read_csv('feature_extraction/features/vgg16_feature.csv')
features_df.shape

# load VGG19 model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

base_model = VGG19(weights='imagenet', include_top=False)
base_model.summary()
# specify the VGG19 layer from which features are extracted
# you can choose not to use the 3 lines below, then by default it will extract the features of
# the last dense layer before softmax classification layer
layer_name = 'block1_conv2'
model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
model.summary()
# specify input image size
img_h, img_w = 224, 224
# iterate through plots and extract features
start = time.time()
with open(current_dir + '/feature_extraction/features/vgg19_feature.csv', 'w') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    for file_name in os.listdir(current_dir + '/feature_extraction/plots/'):
        print(file_name)
        img = image.load_img(current_dir + '/feature_extraction/plots/'+file_name, target_size=(img_h, img_w))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        feature = model.predict(img_data)
        # flatten the output feature and write it to csv file
        feature_np = np.array(feature)
        row = list(feature_np.flatten())
        row.append(file_name)
        writer.writerow(row)
end = time.time()
print('Time taken:', end - start)

features_df = pd.read_csv('feature_extraction/features/vgg19_feature.csv')
features_df.shape

###########################################
### feature extraction from InceptionV3 ###
###########################################
"""
Using InceptionV3 Pre-trained Models for Feature Extraction
Default input size: 299*299 RGB image
Layer_name: specify the layer from which features are extracted
Output: features extracted (flattened) from the specified layer
"""
# load InceptionNet V3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

base_model = InceptionV3(weights='imagenet', include_top=False)
base_model.summary()

model = base_model
# specify input image size
img_h, img_w = 299, 299
# iterate through plots and extract features
start = time.time()
with open(current_dir + '/feature_extraction/features/inception_v3_feature.csv', 'w') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    for file_name in os.listdir(current_dir + '/feature_extraction/plots/'):
        print(file_name)
        img_path = 'feature_extraction/plots/'+file_name
        img = image.load_img(img_path, target_size=(img_h, img_w))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        feature = model.predict(img_data)
        print(feature.shape)
        feature_np = np.array(feature)
        row = list(feature_np.flatten())
        row.append(file_name)
        writer.writerow(row)
end = time.time()
print('Time taken:', end - start)


features_df = pd.read_csv('feature_extraction/features/inception_v3_feature.csv', header=None)
features_df.set_index(131072, inplace=True)
features_df.shape
np.array(features_df.iloc[0]).reshape(8, 8, 2048)
np.sqrt(features_df.shape[1] - 1)
