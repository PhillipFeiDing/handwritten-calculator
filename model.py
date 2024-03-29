import matplotlib.pyplot as plt
import numpy as np
from mark_objects import mark_objects, square_objects
from extract_features import resize, plot_feature, feature, extract_features
from train_classifier import model_save_path, load_clf
from pca_dim_reduction import load_pca, pca_save_path
from image_utils import convolution, scale_color, get_negative, gray2rgb, pooling, rgb2gray


class Model:

    def __init__(self):
        self._classifier = load_clf(model_save_path)
        self._pca = load_pca(pca_save_path)
        self._conv_kernel = np.array([
            [0.2, 0.8, 1.0, 0.8, 0.2],
            [0.8, 1.6, 2.0, 1.6, 0.8],
            [1.0, 2.0, 3.0, 2.0, 1.0],
            [0.8, 1.6, 2.0, 1.6, 0.8],
            [0.2, 0.8, 1.0, 0.8, 0.2]
        ])
        self._pool_x = 3
        self._pool_y = 3
        self._white_threshold=200
        self._scale_factor = np.array([1.0] * 180 + [2.0] * 76, dtype=float)
        self._min_ratio = 0.4

    def predict(self, img):

        conv_img = img
        conv_img = get_negative(conv_img)

        conv_img = convolution(conv_img, self._conv_kernel, gray_scale=False)
        conv_img = rgb2gray(conv_img)
        conv_img = pooling(conv_img, self._pool_x, self._pool_y, method="mean")
        conv_img = gray2rgb(conv_img, a=True)

        conv_img[conv_img > 255] = 255
        conv_img = get_negative(conv_img)
        conv_img = scale_color(conv_img, lightness_factor=self._scale_factor)
        conv_img = np.uint8(conv_img)

        plt.imshow(conv_img)
        plt.show()

        objects = mark_objects(conv_img, min_ratio=self._min_ratio, white=self._white_threshold)
        objects.sort(key=lambda object_pos: object_pos[1])

        squared_img = square_objects(conv_img, objects)
        plt.imshow(squared_img)
        plt.show()
        # plt.imshow(conv_img)
        # plt.show()

        # plt.imsave("conv_img.png", conv_img)

        features = extract_features(conv_img, objects)

        # for i in range(features.shape[0]):
        #     feature = features[i, :]
        #     plot_feature(feature)

        features = self._pca.transform(features)

        labels = self._classifier.predict(features)

        return objects, np.array(labels)


if __name__ == "__main__":
    path = "sample_expressions/"
    test_img_name = input("Path to image: " + path)

    model = Model()

    objects, labels = model.predict(plt.imread(path + test_img_name, format="rgba"))

    for i in range(len(objects)):
        print("%s: %d" % (str(objects[i]), labels[i]))
