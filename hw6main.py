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
        outpath = path.split('.')[0].split('/')[-1]
        out_dir = os.path.join(output, outpath)
        img = readImgCV(path)
        gray = bgr2gray(img)

        img = gauss_blur(img)
        gray = gauss_blur(gray)

        hist = histogram(gray, bins)
        '''
        Otsu on Gray scale
        '''
        thresh, thresh_inv = otsu(gray, bins, hist)
        name = "otsu_gray.jpg"
        save_img_v2(name, out_dir, thresh_inv)
        # cv2show(thresh_inv, "otsu_gray")

        cnt1 = contour(thresh_inv)
        name = "otsu_gray_countour.jpg"
        save_img_v2(name, out_dir, cnt1)
        # cv2show(cnt1, "otsu_gray_countour")

        '''
        Otsu on RGB Image
        '''
        thresh_rgb, thresh_rgb_inv = otsu_rgb(img, bins)
        name = "otsu-r.jpg"
        save_img_v2(name, out_dir, thresh_rgb_inv[:,:,0])
        # cv2show(thresh_rgb_inv[:,:,0], "otsu-r")
        name = "otsu-g.jpg"
        save_img_v2(name, out_dir, thresh_rgb_inv[:,:,1])
        # cv2show(thresh_rgb_inv[:,:,1], "otsu-g")
        name = "otsu-b.jpg"
        save_img_v2(name, out_dir, thresh_rgb_inv[:,:,2])
        # cv2show(thresh_rgb_inv[:,:,2], "otsu-b")

        combined_img_inv = channel_and(thresh_rgb_inv)
        name = "otsu_combined_rgb.jpg"
        save_img_v2(name, out_dir, combined_img_inv)
        # cv2show(combined_img_inv, "otsu_combined_rgb")

        '''
        Combined Contour
        '''
        cnt2 = contour(combined_img_inv)
        name = "otsu_rgb_contour.jpg"
        save_img_v2(name, out_dir, cnt2)
        # cv2show(cnt2, "otsu_rgb_contour")

        '''
        Texture Based Otsu
        '''
        texture_img = otsu_texture(gray, window_size)
        texture_img_rgb, texture_img_rgb_inv = otsu_rgb(texture_img, bins)
        name = "otsu-win_3_texture.jpg"
        save_img_v2(name, out_dir, texture_img_rgb_inv[:, :, 0])
        # cv2show(texture_img_rgb_inv[:, :, 0], "otsu-win_3_texture")
        name = "otsu-win_5_texture.jpg"
        save_img_v2(name, out_dir, texture_img_rgb_inv[:, :, 1])
        # cv2show(texture_img_rgb_inv[:, :, 1], "otsu-win_5_texture")
        name = "otsu-win_7_texture.jpg"
        save_img_v2(name, out_dir, texture_img_rgb_inv[:, :, 2])
        # cv2show(texture_img_rgb_inv[:, :, 2], "otsu-win_7_texture")

        combined_img_texture_inv = channel_and(texture_img_rgb_inv)
        name = "otsu_combined_texture.jpg"
        save_img_v2(name, out_dir, combined_img_texture_inv)
        # cv2show(combined_img_texture_inv, "otsu_combined_texture")

        '''Contour here'''
        cnt3 = contour(combined_img_texture_inv)
        name = "otsu_texture_contour.jpg"
        save_img_v2(name, out_dir, cnt3)
        # cv2show(cnt3, "otsu_texture_contour")
if __name__=='__main__':
    main()