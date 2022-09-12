import configparser
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from vision import *

config = configparser.ConfigParser()
config.read('hw3config.txt')

def main():
    x_img_path = os.path.join(config['PARAMETERS']['top_dir'], config['PARAMETERS']['x_path'])
    x_pts = config['PARAMETERS']['x_pts']
    x_prime_pts = config['PARAMETERS']['x_prime_pts']
    homo_mode = config['PARAMETERS']['homo_mode']
    x_img = readImgCV(x_img_path)
    ho, wo, co = x_img.shape
    hp,wp,cp = ho, wo, co
    x_pts = str2np(x_pts)
    # x_prime_pts=str2np(x_prime_pts)
    x_prime_pts = np.array([(0, 0), (wo, 0), (0, ho), (wo, ho)])
    vision = Vision(x_pts, x_prime_pts)
    h = vision.calc_homograpy(homo_mode)
    # h=np.linalg.inv(h)
    empty_img = np.ones((hp, wp, cp), np.uint8)
    for c in tqdm(range(wo)):
        for r in range(ho):
            X = np.array((c, r, 1))
            X_prime = np.matmul(h, X)
            X_prime = X_prime / X_prime[2]
            X_prime = X_prime.astype(np.int32)
            if X_prime[1] < hp and X_prime[0] < wp and X_prime[0]>0 and X_prime[1]>0:
                empty_img[X_prime[1]][X_prime[0]] = x_img[r][c]
    plt.imshow(empty_img)
    plt.show()

if __name__=='__main__':
    main()