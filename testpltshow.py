## read_img.py

import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

source_path='E:/CToE/Pictures'

def show_pic():
    # img=cv2.imread(os.path.join(source_path,'t1.png'))
    img=Image.open(os.path.join(source_path,'1.png'))
    #   plt.imshow(img)
    info = plt.imshow(img)
    print(info)
    plt.show()

if __name__ == '__main__':
    show_pic()