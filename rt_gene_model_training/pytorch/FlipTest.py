import numpy as np
import cv2

img = cv2.imread("../rt_gene_dataset/s001_glasses/inpainted/left/left_000004_rgb.png")

print(img)
img = np.fliplr(img)

print("=============================================")
print((img))

cv2.imwrite("./fliptest.jpg",img)