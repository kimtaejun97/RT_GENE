import cv2
import os

img_path = "../rt_gene_dataset/s000_glasses/original/face_before_inpainting"
label_path = "../rt_gene_dataset/s000_glasses/label_combined.txt"
img_list = os.listdir(img_path)

file_name = []
with open(label_path,"r") as f:
    #read header
    line = f.readline()

    for line in f:
        line = line.split(",")
        file_name.append(f'{line[0]:0>6}')


count =0
# for img_name in img_list:
#     img_num = img_name.split(".")
#
#     if img_num[0] not in file_name:
#         count+=1
#         os.remove(os.path.join(img_path, img_name))

for img_name in img_list:
    img_num = img_name.split("_")

    if img_num[1] not in file_name:
        count+=1
        os.remove(os.path.join(img_path, img_name))