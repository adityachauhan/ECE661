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
    calibration_pattern_path = glob.glob(top_dir+'/*.png')
    calibration_pattern = readImgCV(calibration_pattern_path[0])
    calibration_pattern_mode = "canny+hough"
    calibration_pattern_edges = cannyEdge(calibration_pattern)
    calibration_pattern_lines = houghLinesP(calibration_pattern_edges)
    plotLinesP(calibration_pattern_lines, calibration_pattern)
    calibration_pattern_corners = findCorner(calibration_pattern_lines, calibration_pattern)
    calibration_pattern_refined_pts, calibration_pattern_num_pts = refinePts(calibration_pattern_corners,20)
    # calibration_pattern_refined_pts, calibration_pattern_num_pts = refinePts(calibration_pattern_refined_pts,10)
    # if calibration_pattern_num_pts < 80:
    #     calibration_pattern_harrisCor, calibration_pattern_corImg = cv2HarrisCorner(calibration_pattern)
    #     calibration_pattern_refined_pts, calibration_pattern_num_pts = refinePts(calibration_pattern_harrisCor, 10)
    #     calibration_pattern_mode = "harris"
    print(calibration_pattern_num_pts)
    calibration_pattern = plotPoints(calibration_pattern_refined_pts, calibration_pattern, mode=calibration_pattern_mode)
    # cv2show(calibration_pattern, "corners")

    for i in range(5):
        img = readImgCV(image_paths[i])
        orig_img = np.copy(img)
        img_edges = cannyEdge(img)
        lines = houghLinesP(img_edges)
        corners = findCornerP(lines, img)
        refined_pts, num_pts = refinePtsP(corners,20)
        # refined_pts, num_pts = refinePts(refined_pts,10)
        mode = "canny+hough"
        # if num_pts < 80:
        #     harrisCor, corImg = cv2HarrisCorner(orig_img)
        #     refined_pts, num_pts = refinePts(harrisCor, 10)
        #     mode = "harris"
        print(num_pts)
        orig_img = plotPoints(refined_pts, orig_img, mode=mode)
        # cv2show(orig_img, "corners")


if __name__ == '__main__':
    main()