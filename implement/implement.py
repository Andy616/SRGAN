import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv
import sys
import os
import re

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

def content_loss(hr_images, sr_images):
    net = MobileNetV2(input_shape=target_image_shape,include_top=False, weights='imagenet')
    net.trainable=False
    for i in net.layers:
        i.trainable=False
    model = Model(inputs=net.input,outputs=net.get_layer('block_4_expand').output)
    model.trainable=False
    loss = tf.losses.mse(model(hr_images), model(sr_images))
    return loss

def bilateral_filter(image):
    return cv.bilateralFilter(image, 0, 10, 1)

model_path = './gen_model100.h5'
model = tf.keras.models.load_model(model_path, custom_objects={ 'content_loss': content_loss })

dirs = os.listdir()
images = []
image_formats = {'jpg', 'jpeg'}
for i in dirs:
    match = re.findall('.+\.(.+)', i)
    if len(match)==1 and match[0] in image_formats:
        images.append(i)
        
def up_scaling(file_name):
    pic = plt.imread('./%s' % file_name)
    pp = model.predict(np.array([pic/255]))
    ps = cv.cvtColor(pp[0],cv.COLOR_RGB2BGR)
    p = bilateral_filter(ps)
    cv.imwrite('%s_upscaled.png' % file_name.split('.')[0], p*255)
    
if __name__=='__main__':
    for image in images:
        up_scaling(image)