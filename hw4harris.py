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
    img_path = os.path.join(config['PARAMETERS']['top_dir'], config['PARAMETERS']['img_path3'])
    filter = config['PARAMETERS']['filter']
    img_orig = readImgCV(img_path)
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)
    img_gray=img_gray/255.0
    h,w=img_gray.shape
    pts = harrisCornerDetector(img_gray, filter=filter, sigma=1.2)
    for i in range(len(pts[0])):
        cv2.circle(img_orig, (pts[1][i], pts[0][i]), radius=1, color=(255,0,0), thickness=-1)

    plt.imshow(img_orig)
    plt.show()
    # cv2.imshow("img", img_orig)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()