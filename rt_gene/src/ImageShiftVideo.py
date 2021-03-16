import cv2
import numpy as np
import os


img_path = "./samples_gaze/003651.png"
out_path = "./samples_video/NoCropLeftShift_3651.mp4"







# padding_img = np.zeros((height, 1000+width,3))
# padding_img[0:height,1:width+1,:] = img
# padding_width = len(padding_img[0])
# padding_height = len(padding_img)
# print(padding_width, padding_height)

img = cv2.imread(img_path)

#오른쪽에 패딩 1000
# npad = ((10,1),(10,1000),(0,0))

#왼쪽에 패딩 1000
npad = ((10,1),(1000,10),(0,0))

img = np.pad(img,npad,'constant',constant_values=(0))

height = len(img)
width = len(img[0])


fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 코덱 정의
out_video = cv2.VideoWriter(out_path, fourcc, 30, (width,height))  # VideoWriter 객체 정의

#x축 방향으로 1px 이동변환 매트릭스
#[[1,0,x],[0,1,y]]
shift = np.float32([[1,0,-1],[0,1,0]])
for i in range(width):
    out_video.write(img)
    print(f'Write {i}frame')
    img = cv2.warpAffine(img,shift,(width,height))


out_video.release()