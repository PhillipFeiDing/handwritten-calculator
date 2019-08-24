import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from math import sqrt
from math import pow
from image_utils import rgb2gray


white_threshold = 240 # 0-255
marker_color = np.array([255, 0, 0], dtype=int)
outlier_threshold = 0.1
min_height_width_ratio = 1.2
max_height_width_ratio = 2
min_height_pixel = 5
min_width_pixel = 5
min_merge_overlap = 0.3


def mark_objects(img, min_ratio=outlier_threshold, white=white_threshold, criterion="diagonal", reference="max", merge=False):
    img = rgb2gray(img)

    content = img < white
    visited = np.full(shape=content.shape, fill_value=False)
    objects = []

    for x in range(content.shape[0]):
        for y in range(content.shape[1]):
            object_pos = test_object(img, content, visited, x, y)
            if object_pos:
                objects.append(object_pos)

    if merge:
        objects = merge(objects)

    objects = filter(objects, min_ratio, criterion, reference)

    return objects


def common_area(rec1, rec2):
    height = overlap_interval(rec1[0], rec1[2], rec2[0], rec2[2])
    width = overlap_interval(rec1[1], rec1[3], rec2[1], rec2[3])

    return width * height


def overlap_interval(range1l, range1r, range2l, range2r):
    if range1l > range2l:
        return overlap_interval(range2l, range2r, range1l, range1r)

    temp = min(range1r, range2r) - max(range1l, range2l)
    if temp > 0:
        return temp
    else:
        return 0


def filter(objects, min_ratio=outlier_threshold, criterion="diagonal", reference="max"):

    accepted_criterion = ["diagonal", "area"]
    assert criterion in accepted_criterion, "criterion must be in %s" % str(accepted_criterion)

    accepted_reference = ["max", "mean", "median"]
    assert reference in accepted_reference, "reference must be in %s" % str(accepted_reference)

    if criterion == "diagonal":
        values = np.array([sqrt(pow(right- left + 1, 2) + pow(bottom - top + 1, 2)) for top, left, bottom, right in objects])
    else:
        values = np.array([(bottom - top) * (right - left) for top, left, bottom, right in objects])

    if values.size < 2:
        return objects

    if reference == "max":
        ref_value = np.max(values)
    elif reference == "mean":
        ref_value = np.mean(values)
    else:
        ref_value = np.median(values)

    ret = []
    for i in range(values.size):
        if values[i] > min_ratio * ref_value:
            ret.append(objects[i])

    return ret


def square_objects(img, objects):
    img = img.copy()

    for object_pos in objects:
        # draw top and bottom
        for y in range(object_pos[1], object_pos[3] + 1):
            img[object_pos[0], y, :3] = marker_color
            img[object_pos[2], y, :3] = marker_color
        # draw left and right
        for x in range(object_pos[0], object_pos[2] + 1):
            img[x, object_pos[1], :3] = marker_color
            img[x, object_pos[3], :3] = marker_color

    return img


def test_object(img, content, visited, x, y):
    if visited[x, y] or not content[x, y]:
        return None

    queue = deque([])
    dir_x = [-1, 0, 1, 0] # top, left, bottom, right
    dir_y = [0, -1, 0, 1]
    top, left, bottom, right = x, y, x, y

    queue.append((x, y))
    while len(queue) > 0:
        curr_x, curr_y = queue.popleft()
        if visited[curr_x, curr_y]:
            continue
        visited[curr_x, curr_y] = True
        top, left, bottom, right = min(curr_x, top), min(curr_y, left), max(curr_x, bottom), max(curr_y, right)
        for i in range(4):
            next_x = curr_x + dir_x[i]
            next_y = curr_y + dir_y[i]
            if not get_content(content, next_x, next_y) or visited[next_x, next_y]:
                continue
            queue.append((next_x, next_y))

    return adjust_square(top, left, bottom, right, content.shape[0], content.shape[1])


def adjust_square(top, left, bottom, right, img_height, img_width):
    height = bottom - top
    width = right - left

    if (height < min_height_pixel and width < min_width_pixel) or height == 0 or width == 0:
        return None

    height_width_ratio = height / width
    if height_width_ratio < min_height_width_ratio:
        # increase height
        increment = int(np.ceil((width * min_height_width_ratio - height) / 2))
        top = top - increment
        bottom = bottom + increment
    elif height_width_ratio > max_height_width_ratio:
        # increase width
        increment = int(np.ceil((height / max_height_width_ratio - width) / 2))
        left = left - increment
        right = right + increment

    return max(0, top), max(0, left), min(img_height - 1, bottom), min(img_width - 1, right)


def get_content(content, x, y):
    if not (0 <= x < content.shape[0] and 0 <= y < content.shape[1]):
        return False
    return content[x, y]


if __name__ == "__main__":
    file_name = "sample_expressions/79236.png"
    img = plt.imread(file_name, format="rgba")
    objects = mark_objects(img)
    img = square_objects(img, objects)
    plt.imshow(img)
    plt.show()
