import os
import numpy as np
import cv2 as cv

source_path = './CelebA/Img/img_align_celeba/'

hr_save_path = './preprocessed_celeb/high_resolution/'
lr_save_path = './preprocessed_celeb/low_resolution/'

if not os.path.exists(hr_save_path):
    os.mkdir(hr_save_path)
if not os.path.exists(lr_save_path):
    os.mkdir(lr_save_path)

for i in range(1,5001):
    x = cv.imread(f'{source_path}{str(i).zfill(6)}.jpg')[2:,2:,:]
    y = cv.resize(x,(44,54))
    cv.imwrite(f'{hr_save_path}{i}.jpg',x)
    cv.imwrite(f'{lr_save_path}{i}.jpg',y)