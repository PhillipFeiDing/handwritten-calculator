from image_utils import rgb2gray
import numpy as np
from cv2 import imread, imwrite


min_row_size = 10
min_col_size = 10
max_row_size = 30
max_col_size = 30

white = 200


def bool_intervals(img, white_threshold=white):
    img = rgb2gray(img)

    row_vec = np.full(shape=img.shape[0], fill_value=True)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] > white_threshold:
                row_vec[x] = False
                break

    col_vec = np.full(shape=img.shape[1], fill_value=True)
    for y in range(img.shape[1]):
        for x in range(img.shape[0]):
            if img[x, y] > white_threshold:
                col_vec[y] = False
                break

    return row_vec, col_vec


def draw_grid(img, color=np.array([0, 255, 255])):
    row_vec, col_vec = bool_intervals(img)

    img = img.copy()
    img[row_vec, :, 0] = color[0]
    img[row_vec, :, 1] = color[1]
    img[row_vec, :, 2] = color[2]

    img[:, col_vec, 0] = color[0]
    img[:, col_vec, 1] = color[1]
    img[:, col_vec, 2] = color[2]

    row_count = 0
    curr_size = 0
    for i in range(len(row_vec)):
        if row_vec[i]:
            curr_size = 0
        else:
            curr_size = curr_size + 1

        if curr_size == min_row_size:
            row_count = row_count + 1
        elif curr_size > max_row_size:
            print("warning: may missed one or more rows")


    col_count = 0
    curr_size = 0
    for i in range(len(col_vec)):
        if col_vec[i]:
            curr_size = 0
        else:
            curr_size = curr_size + 1

        if curr_size == min_col_size:
            col_count = col_count + 1
        elif curr_size == max_col_size:
            print("warning: may missed one or more cols")

    return img, row_count, col_count


if __name__ == "__main__":

    img_path = "train_image/mnist_train9_conv.jpg"
    save_path = "train_image/mnist_train9_grid.jpg"

    img = imread(img_path)

    grid_img, row, col = draw_grid(img)
    imwrite(save_path, grid_img)
    print(row, "*", col)