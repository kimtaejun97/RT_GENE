from scipy import io
import numpy as np
# import tables
# file = tables.open_file("../mtcnn_twoeyes_inpainted_eccv/RT_GENE_train_s000.mat")

mat_file = io.loadmat("../mtcnn_twoeyes_inpainted_eccv/RT_GENE_train_s002.mat")
#[train label][value label][del array][del array]
gaze = np.vstack(mat_file['train']['gazes'][0][0])
headpose = np.stack(mat_file['train']['headposes'][0][0])
image_l = np.stack(mat_file['train']['imagesL'][0][0])
image_r = np.stack(mat_file['train']['imagesL'][0][0])
print("head pose: \n",headpose[0][0])
print("gaze:\n",gaze)
print(image_l, len(image_l))
print(image_r)


