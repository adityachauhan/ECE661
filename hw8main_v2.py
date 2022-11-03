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

    for i in range(1):
        img = readImgCV(image_paths[i])
        orig_img1 = np.copy(img)
        orig_img2 = np.copy(img)
        orig_img3 = np.copy(img)
        img_edges = cannyEdge(img)
        lines = houghLines(img_edges)
        img = plotLines(lines, img)
        cv2show(img, "img")
        l1, l2 = filterLines(lines)
        orig_img1 = plotLines(l1, orig_img1)
        cv2show(orig_img1, "origImg")
        refineLines(l1)
        # orig_img2 = plotLines(l2, orig_img2)
        # cv2show(orig_img2, "origImg")
        # refined_l1 = clubLines(l1, img)
        # orig_img3 = plotLines(refined_l1, orig_img3)
        # cv2show(orig_img3, "or3")

        # img_lines = plotLines(lines, img)
        # cv2show(img_lines,"lines")
        # corners = findCorner(lines, img)
        # print(len(corners))
        # refined_pts, num_pts = refinePts(corners, 19)
        # print(num_pts)
        # plotPoints(refined_pts, orig_img, "canny+hough")
        # cv2show(orig_img, "img")


if __name__ == '__main__':
    main()