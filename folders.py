import os
import cv2
import numpy as np

images_path = r'/media/taher/24288281288251AA/datasets/kh/images/train'
labels_path = r'/media/taher/24288281288251AA/datasets/kh/labels/train'

images_list = os.listdir(images_path)
labels_list = os.listdir(labels_path)

ys = []

# for ind, label_name in enumerate(labels_list):
#     image_name = images_list[ind]
#     img = cv2.imread(f'/media/taher/24288281288251AA/datasets/kh/images/train/{image_name}')
file = open(f'/media/taher/24288281288251AA/datasets/kh/labels/train/{labels_list[0]}')

image_name = images_list[0]
img = cv2.imread(f'/media/taher/24288281288251AA/datasets/kh/images/train/{image_name}')
for line in file.readlines():
    wh_list = list(map(float, line[:-2].split()))
    label = wh_list[0]
    ys.append(label)
    h = list(map(lambda x: int(x * 240), wh_list[1:3]))
    w = list(map(lambda x: int(x * 320), wh_list[3:]))
    # print([h[0], h[1], w[0], w[1]])
    try:
        cv2.imshow('image', img[h[0]: h[1], w[0]: w[1]])
        print(label, h, w)
    except:
        pass
    cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
