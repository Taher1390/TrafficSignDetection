import os
import cv2
import numpy as np

images_path = '/home/taher/projects/Traffic-Signs-Dataset-YOLOv8-YOLOv11/train/images'
labels_path = '/home/taher/projects/Traffic-Signs-Dataset-YOLOv8-YOLOv11/train/labels'
save_path = '/home/taher/projects/Traffic-Sign-Detection/dataset/noentry'

def read_file(file_name):
    file = open(f'{labels_path}/{file_name}', 'r')
    return file.read()

images_names = os.listdir(images_path)
labels_names = os.listdir(labels_path)

for i, image_name in enumerate(images_names):
    image = cv2.imread(f'{images_path}/{image_name}')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:-1]
    text = list(map(float, read_file(image_name[:-3]+'txt').split()))

    target = text[0]
    x_c, y_c = list(map(lambda x: int(x * h), text[1:3]))
    w_tar, h_tar = list(map(lambda x: int(x * h), text[3:]))
    if target == 4:
        x1 = x_c - w_tar//2
        y1 = y_c - h_tar//2
        x2 = x_c + w_tar//2
        y2 = y_c + h_tar//2
        crop = image[y1: y2, x1: x2]
        
        cv2.imwrite(f'{save_path}/{image_name}', crop)

        # cv2.imshow(image_name, crop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # break
