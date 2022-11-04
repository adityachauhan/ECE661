import random
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
from einops import rearrange

config = configparser.ConfigParser()
config.read('hw8config.txt')

def main():
    top_dir = config['PARAMETERS']['top_dir']
    data_dir = config['PARAMETERS']['data_dir']
    data_dir = os.path.join(top_dir, data_dir)
    image_paths = glob.glob(data_dir+'/*.jpg')
    num_images = len(image_paths)

    for i in range(5):
        img = readImgCV(image_paths[i])
        orig_img1 = np.copy(img)
        orig_img2 = np.copy(img)
        orig_img3 = np.copy(img)
        orig_img4 = np.copy(img)
        img_edges = cannyEdge(img)
        lines = houghLines(img_edges)
        img = plotLines(lines, img)
        cv2show(img, "img")
        l1, l2, type = filterLines(lines)
        orig_img1 = plotLines(l1, orig_img1)
        cv2show(orig_img1, "origImg")
        if type == "v":
            refined_l1 = refineLines(l1, "v", 8)
            orig_img2 = plotLines(refined_l1, orig_img2)
            cv2show(orig_img2, "origImg")
            orig_img3 = plotLines(l2, orig_img3)
            cv2show(orig_img3, "origImg")
            refined_l2 = refineLines(l2, "h", 10)
            orig_img4 = plotLines(refined_l2, orig_img4)
            cv2show(orig_img4, "origImg")
        elif type == "h":
            refined_l1 = refineLines(l1, "h", 10)
            orig_img2 = plotLines(refined_l1, orig_img2)
            cv2show(orig_img2, "origImg")
            orig_img3 = plotLines(l2, orig_img3)
            cv2show(orig_img3, "origImg")
            refined_l2 = refineLines(l2, "v", 8)
            orig_img4 = plotLines(refined_l2, orig_img4)
            cv2show(orig_img4, "origImg")


if __name__ == '__main__':
    main()