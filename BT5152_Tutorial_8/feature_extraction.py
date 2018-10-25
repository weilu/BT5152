import os
import numpy as np
import pandas as pd
from keras.models import Model
from keras.preprocessing import image

# Specify the pre-trained CNN model
ALGORITHM = "ResNet50"

if ALGORITHM == "ResNet50":
    from keras.applications.resnet50 import ResNet50 as Model
    from keras.applications.resnet50 import preprocess_input, decode_predictions
    img_h, img_w = 224, 224
elif ALGORITHM == "VGG19":
    from keras.applications.vgg19 import VGG19 as Model
    from keras.applications.vgg19 import preprocess_input
    img_h, img_w = 224, 224
elif ALGORITHM == "InceptionV3":
    from keras.applications.inception_v3 import InceptionV3 as Model
    from keras.applications.inception_v3 import preprocess_input
    img_h, img_w = 299, 299
else:
    raise ValueError('{} is not a valid choice.'.format(ALGORITHM))


model = Model(weights='imagenet', include_top=False) # include_top=False to change it to feature extraction

data = {
    'trump': [],
    'putin': [],
    'test': []
}

for root, directory, files in os.walk(os.path.join(os.getcwd(), 'feature_extraction/data')):
    for f in files:
        label = root.split('/')[-1]
        img = image.load_img(os.path.join(root, f), target_size=(img_h, img_w))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        feature = model.predict(img_data)
        feature_np = np.array(feature)
        data[label].append(list(feature_np.flatten()))

# You can check the dimensions of the variables by uncommenting these lines
# img_data.shape
# feature.shape

for k, preds in data.items():
    for pred in preds:
        tops = decode_predictions(np.array(pred).reshape(1, -1), top=3)[0]

        for top in tops:
            print("{}: {}".format(k, top))

# Use pandas to look at the data and merge the data frames
# You can use the data directly to feed into the algorithm without transforming
# to DataFrame. This is more for looking at the data.

# Create empty DataFrame
df = pd.DataFrame()

# This is generally not recommended for large dataset as appending new data to
# DataFrame is very slow. For demo purpose only.
for label, datum in data.items():
    temp_df = pd.DataFrame(datum)
    temp_df['label'] = label
    df = df.append(temp_df, ignore_index=True)

# Get the label for each data; remove test data
y = df[df.label != 'test'].label
# Get all the features and remove the label
X = df[df.label != 'test'].drop('label', axis=1)

# From here onwards, you can use any ML algorithm for the training on the
# extracted features
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# Near zero variance filter
nzv_filter = VarianceThreshold()
# Multi-layer perceptron model, this is equivalent to the fully connected layers
# in the CNN model
mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 10))

# Using Pipeline to chain processes
mlp_pipeline = Pipeline([('nzv', nzv_filter), ('mlp', mlp_clf)])

# Train the model
mlp_pipeline.fit(X, y)

# Do prediction on test data
mlp_pipeline.predict(data['test'])
