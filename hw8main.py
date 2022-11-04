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
    out_dir = config['PARAMETERS']['output']
    image_paths = glob.glob(data_dir+'/*.jpg')
    num_images = len(image_paths)


    for i in range(num_images):
        img = readImgCV(image_paths[i])
        orig_img = np.copy(img)
        img_edges = cannyEdge(img)
        lines = houghLines(img_edges)
        l1, l2, type = filterLines(lines)
        if type == "v":
            refined_l1 = refineLines(l1, "v", 8)
            img = plotLines(refined_l1, img, (255,0,0))
            refined_l2 = refineLines(l2, "h", 10)
            img = plotLines(refined_l2, img, (0,255,0))
            corners = getCorners(refined_l1, refined_l2)
            plotPoints(corners, orig_img, mode="canny+hough")
        elif type == "h":
            refined_l1 = refineLines(l1, "h", 10)
            img = plotLines(refined_l1, img, (0,255,0))
            refined_l2 = refineLines(l2, "v", 8)
            img = plotLines(refined_l2, img, (255,0,0))
            corners = getCorners(refined_l2, refined_l1)
            plotPoints(corners, orig_img, mode="canny+hough")

        houghName = "houghLines_"+str(i+1)+'.png'
        save_img(houghName, out_dir, img)
        name = "corners_"+str(i+1)+'.png'
        save_img(name, out_dir, orig_img)



if __name__ == '__main__':
    main()