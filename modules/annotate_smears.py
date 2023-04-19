import cv2
import os
import time
from typing import List


# Constant parameters
color_code = {
    'unhealthy': (28, 32, 190),  # red
    'healthy': (100, 131, 54),  # green
    'rubbish': (204, 102, 0),  # blue
    'bothcells': (136, 40, 81)  # purple
}
class_code = {
    0: 'healthy',
    1: 'rubbish',
    2: 'unhealthy',
    3: 'bothcells'
}


def plot_box(box_indices: List[float], image: object, color: tuple[int], class_name: str):
    """
    Plots a single colored and labeled box onto the image.

    :param box_indices: list of 4 indices, in an [x1, y1, x2, y2], where [x1, y1] are coordinated of the starting
    point of the rectangle and [x2, y2] are coordinates of the end point of the rectangle
    :param image: image to be annotated
    :param color: color of the box, based on the class of the cell
    :param class_name: name of the class of the cell
    """
    line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1

    # plot annotation box
    c1, c2 = (int(box_indices[0]), int(box_indices[1])), (int(box_indices[2]), int(box_indices[3]))
    cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

    # plot text box and text
    font_thickness = max(line_thickness - 5, 1)
    t_size = cv2.getTextSize(class_name, 0, fontScale=line_thickness / 5, thickness=font_thickness)[0]
    # c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
    # cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)
    cv2.putText(image, class_name, (c1[0] + 5, c1[1] + t_size[1] + 5), 0, line_thickness / 5,
                color, thickness=font_thickness, lineType=cv2.LINE_AA)


def annotate_smear_images(images_dir: str, labels_dir: str, output_dir: str):
    """
    Annotate every smear image found in the provided in images_dir,
    based on every label text document in the labels_dir.
    :param images_dir: path to the directory where the smear slices are
    :param labels_dir: path to the directory where the labels are
    (same naming should be used as the images, just with .txt at the end, see the example data provided)
    :param output_dir: path to the directory that the annotated smears will be saved in
    """
    im_idx = 0
    start_time = time.time()
    all_im_cnt = len(os.listdir(images_dir))
    for im_name in os.listdir(images_dir):
        im_idx += 1
        if im_idx % 5 == 0:
            print('Processing images: {:6d}/{:6d}'.format(im_idx, all_im_cnt))

        image = cv2.imread(images_dir + im_name)

        height, width, channels = image.shape
        label_doc = open(labels_dir + im_name[:-4] + '.txt')

        for line in label_doc:
            line = line.rstrip()
            txt_data = line.split()
            class_id = int(txt_data[0])

            x_center, y_center, box_width, box_height = float(txt_data[1]) * width, \
                                                        float(txt_data[2]) * height, \
                                                        float(txt_data[3]) * width, \
                                                        float(txt_data[4]) * height

            x1 = round(x_center - box_width / 2)
            y1 = round(y_center - box_height / 2)
            x2 = round(x_center + box_width / 2)
            y2 = round(y_center + box_height / 2)

            plot_box([x1, y1, x2, y2], image, color=color_code[class_code[class_id]], class_name=class_code[class_id])

            cv2.imwrite(output_dir + im_name, image)

        if im_idx == all_im_cnt:
            print('Processing images: {:6d}/{:6d}'.format(im_idx, all_im_cnt))
            print('Processing finished, images saved in: "{}"'.format(output_dir))
            end_time = time.time()
            print('Elapsed time total: {:.2f} min'.format((end_time - start_time) / 60))


annotate_smear_images(images_dir='F:\DECIT\decit-smear-cell-db\data\\',
                      labels_dir='F:\DECIT\decit-smear-cell-db\data\input\labels\\',
                      output_dir='data/output/')
