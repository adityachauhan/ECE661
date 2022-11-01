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
        img_edges = cannyEdge(img)
        lines = houghLinesP(img_edges)
        corners = findCorner(lines, img)
        refined_pts = refinePts(corners,20)
        img = plotPoints(refined_pts, img)
        cv2show(img, "corners")


if __name__ == '__main__':
    main()