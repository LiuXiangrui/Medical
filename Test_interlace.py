import os

from PIL import Image
import numpy as np
import cv2
# cv2.namedWindow("Image")
# cv2.imshow("Image", img)
# cv2.waitKey (0)

from matplotlib import pyplot as plt


folder = r"C:\Users\XiangruiLiu\Desktop\CT_16bit"
even_folder = r"C:\Users\XiangruiLiu\Desktop\CT_even"
odd_folder = r"C:\Users\XiangruiLiu\Desktop\CT_odd"


for img_folder in os.listdir(folder):
    for img in os.listdir(os.path.join(folder, img_folder)):
        img = cv2.imread(os.path.join(folder, img_folder, img), flags=cv2.IMREAD_ANYDEPTH)

        array = np.array(img).astype(np.uint16)
        odd_array = np.zeros_like(array, dtype=np.uint8)
        even_array = np.zeros_like(array, dtype=np.uint8)

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                number = bin(array[i][j])[2:].zfill(16)
                odd_array[i][j] = int("0b" + number[0::2], 2)
                even_array[i][j] = int("0b" + number[1::2], 2)

                odd_number = bin(odd_array[i][j])[2:].zfill(8)
                even_number = bin(even_array[i][j])[2:].zfill(8)
                rec_number = "".join(i + j for i, j in zip(odd_number, even_number))

        even_img = Image.fromarray(even_array)
        even_img.show()
        odd_img = Image.fromarray(odd_array)
        odd_img.show()
        input()