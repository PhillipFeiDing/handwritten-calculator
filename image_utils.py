import numpy as np

default_rgb_weights = np.array([0.2989, 0.5870, 0.1140])


def rgb2gray(img, rgb_weights=default_rgb_weights):
    if img.ndim == 2:
        return img.copy()
    else:
        gray_img = np.array(img[:, :, 0] * rgb_weights[0]
                            + img[:, :, 1] * rgb_weights[1]
                            + img[:, :, 2] * rgb_weights[2], dtype=int)
        gray_img[gray_img > 255] = 255
        gray_img[gray_img < 0] = 0
        return gray_img


def gray2rgb(img, a=False):
    assert img.ndim == 2, "gray scaled image must be 2-d"
    if not a:
        img = np.concatenate([[img], [img], [img]], axis=0)
    else:
        a_layer = np.full(fill_value=255, dtype=int, shape=img.shape[:2])
        img = np.concatenate([[img], [img], [img], [a_layer]], axis=0)
    return img.swapaxes(0, 1).swapaxes(1, 2)


def scale_color(img, lightness_factor=None):
    img = img.copy()
    min_pix = np.min(np.min(np.min(img)))
    max_pix = np.max(np.max(np.max(img)))

    ratio = 255 / (max_pix - min_pix)
    img = np.array(np.round((img - min_pix) * ratio), dtype=int)
    img[img > 255] = 255

    if lightness_factor is not None:
        assert len(lightness_factor) == 256, "The lightness_factor must be of length 256."
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img.ndim == 2:
                    before = img[x, y]
                    img[x, y] = min(int(before * lightness_factor[before]), 255)
                else:
                    for z in range(3):
                        before = img[x, y, z]
                        img[x, y, z] = min(int(before * lightness_factor[before]), 255)

    return np.uint8(img)


def get_negative(img):
    img = img.copy()
    img[:, :, :3] = np.uint8(255) - img[:, :, :3]
    return img


def convolution(img, kernel, gray_scale=True):
    assert img.ndim == 3, "convoluiton takes a rgb(a) image as input, so ndim = 3."
    rgba = img.shape[2]
    img = rgb2gray(img)

    assert kernel.shape[0] <= img.shape[0] and kernel.shape[1] <= img.shape[1], "convolution kernel too large."

    conv_img = np.full(fill_value=0, shape=img.shape)
    shift_x, shift_y = int((kernel.shape[0] - 1) / 2), int((kernel.shape[1] - 1) / 2)

    for x in range(conv_img.shape[0] - kernel.shape[0] + 1):
        for y in range(conv_img.shape[1] - kernel.shape[1] + 1):
            conv_img[x + shift_x, y + shift_y] = np.sum(np.sum(img[x: x + kernel.shape[0], y: y + kernel.shape[1]] * kernel))

    if gray_scale:
        return conv_img
    else:
        return gray2rgb(conv_img, a=rgba == 4)


def pooling(img, pool_x, pool_y, method="mean"):
    assert img.ndim == 2, "pooling takes a gray sacled image as input, so ndim = 2."
    assert pool_x <= img.shape[0] and pool_y <= img.shape[1], "pool_x or pool_y too large."
    accepted_method = ["mean", "max"]
    assert method in accepted_method, "the pooling method must be one of %s" % str(accepted_method)

    new_x = int(img.shape[0] / pool_x)
    new_y = int(img.shape[1] / pool_y)
    pooled = np.full(fill_value=0, shape=(new_x, new_y))

    for x in range(new_x):
        for y in range(new_y):
            sub_img = img[x * pool_x: (x + 1) * pool_x, y * pool_y: (y + 1) * pool_y]
            if method == "mean":
                pooled[x, y] = np.mean(np.mean(sub_img))
            else:
                pooled[x, y] = np.max(np.max(sub_img))

    return pooled


def put_frame(img, target_shape, fill_value=0):
    assert img.shape[0] <= target_shape[0] and img.shape[1] <= target_shape[1], "image too large to be framed."

    framed = np.full(fill_value=fill_value, shape=target_shape)
    start_x = int((target_shape[0] - img.shape[0]) / 2)
    start_y = int((target_shape[1] - img.shape[1]) / 2)
    framed[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1], :] = img
    return framed.copy()
