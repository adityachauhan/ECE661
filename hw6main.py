import numpy as np

from vision import *
import configparser
from tqdm import tqdm
from utility import *
import matplotlib.pyplot as plt
import os
import cv2
import glob
from scipy.optimize import least_squares

config = configparser.ConfigParser()
config.read('hw6config.txt')

def main():
    top_dir = config['PARAMETERS']['top_dir']
    output = config['PARAMETERS']['output']
    bins = int(config['PARAMETERS']['bins'])
    img_paths = glob.glob(top_dir+'/*.jpg')
    window_size = [3,5,7]
    for path in img_paths:
        img = readImgCV(path)
        gray = bgr2gray(img)
        hist = histogram(gray, bins)
        thresh, thresh_inv = otsu(gray, bins, hist)
        cv2show(thresh)
        thresh_rgb = otsu_rgb(img, bins)
        cv2show(thresh_rgb)
        texture_img = otsu_texture(gray, bins, window_size)
        cv2show(texture_img)
        texture_img_rgb = otsu_rgb(texture_img, bins)
        cv2show(texture_img_rgb)
        '''Contour here'''
        cnt1 = contour(thresh)
        cnt2 = contour(thresh_rgb)
        cnt3 = contour(texture_img_rgb)
        cv2show(cnt1)
        cv2show(cnt2)
        cv2show(cnt3)
if __name__=='__main__':
    main()