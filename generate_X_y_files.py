from mark_objects import mark_objects, square_objects
from extract_features import extract_features
from extract_features import feature_number
from image_utils import get_negative
from tqdm import tqdm
import cv2
import numpy as np
import json


min_ratios = [
    0.45, # 0
    0.54, # 1
    0.56, # 2
    0.55, # 3
    0.52, # 4
    0.54, # 5
    0.54, # 6
    0.62, # 7
    0.51, # 8
    0.61  # 9
]
train_X_data_path = "train_data/X_train.json"
train_y_data_path = "train_data/y_train.json"
test_X_data_path = "test_data/X_test.json"
test_y_data_path = "test_data/y_test.json"
train_img_path = ["train_image/mnist_train{}.jpg".format(i) for i in range(10)]
train_img_conv_path = ["train_image/mnist_train{}_conv.jpg".format(i) for i in range(10)]
train_img_squared_path = ["train_image/mnist_train{}_squared.jpg".format(i) for i in range(10)]
test_img_path = ["test_image/mnist_test{}.jpg".format(i) for i in range(10)]
test_img_conv_path = ["test_image/mnist_test{}_conv.jpg".format(i) for i in range(10)]
test_img_squared_path = ["test_image/mnist_test{}_squared.jpg".format(i) for i in range(10)]
labels = list(range(10))


def generate_features_with_label(img, img_conv, save_path, label):
    objects = mark_objects(img_conv, min_ratio=min_ratios[label], white=200)
    squared_img = square_objects(img, objects)
    cv2.imwrite(save_path, squared_img)
    return extract_features(img, objects), np.array([label] * len(objects))


def generate_all_features(img_paths, img_conv_paths, save_paths, labels):
    X = np.array([[0] * feature_number])
    y = np.array([0])
    assert len(img_paths) == len(labels)
    for i in tqdm(range(len(img_paths))):
        img, img_conv = cv2.imread(img_paths[i]), cv2.imread(img_conv_paths[i])
        img, img_conv = get_negative(img), get_negative(img_conv)
        label = labels[i]
        newX, newy = generate_features_with_label(img, img_conv, save_paths[i], label)
        X = np.vstack([X, newX])
        y = np.concatenate([y, newy])
    return X[1:, :], y[1:]


def write_X_y_file(X, y, X_save_path, y_save_path):
    print("\nwriting ...")
    str_X = json.dumps(X.tolist())
    fh = open(X_save_path, 'w')
    fh.write(str_X)
    fh.close()

    str_y = json.dumps(y.tolist())
    fh = open(y_save_path, 'w')
    fh.write(str_y)
    fh.close()


if __name__ == "__main__":
    print("generating train files ...")
    X_train, y_train = generate_all_features(train_img_path, train_img_conv_path, train_img_squared_path, labels)
    write_X_y_file(X_train, y_train, train_X_data_path, train_y_data_path)

    print()

    print("generating test files ...")
    X_test, y_test = generate_all_features(test_img_path, test_img_conv_path, test_img_squared_path, labels)
    write_X_y_file(X_test, y_test, test_X_data_path, test_y_data_path)