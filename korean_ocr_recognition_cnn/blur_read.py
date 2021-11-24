import cv2
import argparse
import glob
import io
import os
import random

import numpy
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt



def generate_hangul_images():


    output_dir = 'C:/blur'
    a = plt.imread('C:/blur/000.jpg')


    image_dir = os.path.join(output_dir, 'blur-result')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))



    print("type a")
    print(a)

    file_path = 'C:/blur/blurring_result'

    #####blurring 코드 부분#####

    arr = numpy.array(a)
    for i in range(1):
        image = cv2.GaussianBlur(arr, (5,5), 0.8)      # kernel = numpy.ones((2,2),numpy.float32)/4##1/25필터 생성
       #image = cv2.filter2D(arr,-1,kernel)## 1/25 필터 적용


    file_string = '{}.jpeg'.format(779)
    file_path = os.path.join(image_dir, file_string)
    image = Image.fromarray(image)
    image.save(file_path, 'JPEG')
    ##########################

generate_hangul_images()