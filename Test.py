import os

from PIL import Image
import numpy as np
import cv2
img = cv2.imread(r"C:\Users\XiangruiLiu\Desktop\CT_16bit\study_0939\0.png", flags=cv2.IMREAD_ANYDEPTH)
# cv2.namedWindow("Image")
# cv2.imshow("Image", img)
# cv2.waitKey (0)

from matplotlib import pyplot as plt


array = np.array(img)

plt.hist(array.flatten(), bins=100)
plt.savefig("mygraph.png")
plt.close()


msb = np.right_shift(np.bitwise_and(array, 0xff00), 8).astype(np.uint8)
lsb = np.bitwise_and(array, 0xff).astype(np.uint8)

plt.hist(msb.flatten(), bins=100)
plt.savefig("msb.png")
plt.close()

plt.hist(lsb.flatten(), bins=100)
plt.savefig("lsb.png")
plt.close()

msb_recover = np.left_shift(msb, 8).astype(np.uint16)
mse = (array - msb_recover) ** 2
mse = mse.mean()
psnr = 10 * np.log10((2 ** 32) / mse)
print(psnr)


msb = Image.fromarray(msb)
msb.show()

lsb = Image.fromarray(lsb)
lsb.show()
exit()
import os

folder = r"C:\Users\XiangruiLiu\Desktop\CT_16bit"

avg_psnr = 0
count = 0

for img_folder in os.listdir(folder):
    for img in os.listdir(os.path.join(folder, img_folder)):
        img = cv2.imread(os.path.join(folder, img_folder, img), flags=cv2.IMREAD_ANYDEPTH)
        array = np.array(img).astype(np.uint16)
        msb = np.right_shift(np.bitwise_and(array, 0xff00), 8).astype(np.uint8)
        msb_recover = np.left_shift(msb, 8).astype(np.uint16)
        mse = (array - msb_recover) ** 2
        mse = mse.mean()
        psnr = 10 * np.log10((2 ** 32) / mse)

        avg_psnr += psnr
        count += 1

avg_psnr = avg_psnr / count
print(avg_psnr)