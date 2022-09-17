import configparser
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from vision import *

config = configparser.ConfigParser()
config.read('hw3config.txt')

def main():
    x_img_path = os.path.join(config['PARAMETERS']['top_dir'], config['PARAMETERS']['x_path'])
    x_pts = config['PARAMETERS']['l_pts']
    x_prime_pts = config['PARAMETERS']['l_prime_pts']
    homo_mode = config['PARAMETERS']['homo_mode']
    x_img = readImgCV(x_img_path)
    ho, wo, co = x_img.shape

    x_pts = str2np(x_pts)
    x_prime_pts=str2np(x_prime_pts)
    hvl = vanishing_line_homography(x_pts)
    print(hvl)
    # x_prime_pts = np.array([(0, 0), (wo, 0), (0, ho), (wo, ho)])
    vision = Vision(x_pts, x_prime_pts)
    h = vision.calc_homograpy(homo_mode)
    h=np.linalg.inv(h)
    eXPrime = xpeqhx(wo,ho,h)
    eXPrime=eXPrime.astype(int)
    hp, wp, cp = eXPrime[1], eXPrime[0], co
    print(hp,wp,cp)
    empty_img = np.ones((hp, wp, cp), np.int32)
    print(empty_img.shape)
    for c in tqdm(range(wp)):
        for r in range(hp):
            X_prime=xpeqhx(c,r,h)
            X_prime=X_prime.astype(int)
            if X_prime[1] < ho and X_prime[0] < wo and X_prime[0]>0 and X_prime[1]>0:
                empty_img[r][c] = x_img[X_prime[1]][X_prime[0]]
    plt.imshow(empty_img)
    plt.show()

if __name__=='__main__':
    main()