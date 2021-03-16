import cv2
import os

img_path = "../rt_gene_dataset/s000_face_SR"
img_list = os.listdir(img_path)


fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 코덱 정의

out_path = os.path.join("./", "s000_face_SR.mp4")
out_video = cv2.VideoWriter(out_path, fourcc, 30, (224,224))  # VideoWriter 객체 정의

# count = 0
for img_name in img_list:
    # count += 1
    # if count>550 and count <1000:
    img = os.path.join(img_path, img_name)
    img = cv2.imread(img,cv2.IMREAD_COLOR)
    out_video.write(img)
    print("write :",img_name)
    # else:
        # continue

out_video.release()