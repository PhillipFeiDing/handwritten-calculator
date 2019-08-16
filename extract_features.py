import cv2
import matplotlib.pyplot as plt
import numpy as np
from image_utils import rgb2gray


max_feature_val = 255 # inclusive
row = 8
col = 6
feature_number = row * col


def extract_features(img, objects):
    X = []
    for object_pos in objects:
        X.append(feature(resize(img, object_pos)))
    return np.array(X)


def resize(img, object_pos):
    top, left, bottom, right = object_pos
    sub_img = img[top: bottom + 1, left: right + 1, :]
    return cv2.resize(sub_img, (col, row), interpolation=cv2.INTER_AREA)


def feature(img):
    img = rgb2gray(img)
    img = 255 - img
    max_val = np.max(np.max(img))

    features = np.full(dtype=int, shape=img.shape[0] * img.shape[1], fill_value=0)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            features[i * img.shape[1] + j] = \
                min(round((img[i, j] / max_val) * max_feature_val), max_feature_val)

    return features


def plot_feature(features):
    plt.matshow(features.reshape(row, col))
    plt.show()