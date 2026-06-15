import os
import cv2
import numpy as np

images_path = r'/home/taher/projects/Traffic-Sign-Detection/dataset/stop'

images_list = os.listdir(images_path)

pictures = []
for image_name in images_list:
    img = cv2.imread(f'/home/taher/projects/Traffic-Sign-Detection/dataset/stop/{image_name}')
    img = cv2.resize(img, (128, 128))
    pictures.append(img)

pictures = np.array(pictures)
print(pictures.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()
