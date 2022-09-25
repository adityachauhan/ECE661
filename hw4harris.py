from vision import *
import configparser
from tqdm import tqdm
from utility import *
import matplotlib.pyplot as plt
import os
import cv2


config = configparser.ConfigParser()
config.read('hw4config.txt')

def main():
    img_path = os.path.join(config['PARAMETERS']['top_dir'], config['PARAMETERS']['img_path1'])
    filter = config['PARAMETERS']['filter']
    img_orig = readImgCV(img_path)
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    h,w=img_gray.shape
    pts = harrisCornerDetector(img_gray, filter=filter, sigma=1.2)
    for pt in pts:
        cv2.circle(img_orig, (pt[0], pt[1]), radius=4, color=(255,0,0), thickness=-1)

    cv2.imshow("img", img_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()