import configparser
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

from vision import *

config = configparser.ConfigParser()
config.read('hw2config.txt')
def main():
    x_img_path = os.path.join(config['PARAMETERS']['top_dir'], config['PARAMETERS']['x_path'])
    x_prime_img_path = os.path.join(config['PARAMETERS']['top_dir'],config['PARAMETERS']['x_prime_path'])
    x_prime_pts = config['PARAMETERS'][config['PARAMETERS']['card_pts']]
    homo_mode = config['PARAMETERS']['homo_mode']
    output = config['PARAMETERS']['output']

    x_img= readImgCV(x_img_path)
    ho, wo, co = x_img.shape
    print(ho,wo,co)
    x_prime_img=readImgCV(x_prime_img_path)
    hp, wp, cp = x_prime_img.shape

    x_prime_pts = str2np(x_prime_pts)
    # x_prime_pts = x_prime_pts.split(',')
    # x_prime_pts = [int(val) for val in x_prime_pts]
    # x_prime_pts = np.array(x_prime_pts)
    # x_prime_pts = rearrange(x_prime_pts, '(c h)-> c h ', c=4, h=2)

    x_pts=np.array([(0,0),(wo,0),(0,ho),(wo,ho)])

    vision = Vision(x_pts, x_prime_pts)
    h = vision.calc_homograpy(homo_mode)

    if config['PARAMETERS']['mapping'] == 'd2r':
        for c in tqdm(range(wo)):
            for r in range(ho):
                X = np.array((c, r, 1))
                X_prime = np.matmul(h, X)
                X_prime = X_prime / X_prime[2]
                X_prime = X_prime.astype(np.int)
                # print(X_prime)
                x_prime_img[X_prime[1]][X_prime[0]] = x_img[r][c]

    if config['PARAMETERS']['mapping'] == 'r2d':
        l12 = np.cross(cvrt2homo(x_prime_pts[0]), cvrt2homo(x_prime_pts[1]))
        l24 = np.cross(cvrt2homo(x_prime_pts[1]), cvrt2homo(x_prime_pts[3]))
        l43 = np.cross(cvrt2homo(x_prime_pts[3]), cvrt2homo(x_prime_pts[2]))
        l31 = np.cross(cvrt2homo(x_prime_pts[2]), cvrt2homo(x_prime_pts[0]))
        hinv = np.linalg.inv(h)
        for c in tqdm(range(wp)):
            for r in range(hp):
                X = np.array((c, r, 1))
                X_prime = np.matmul(hinv, X)
                X_prime = X_prime / X_prime[2]
                X_prime = X_prime.astype(np.int)
                # print(X_prime)
                if np.dot(np.transpose(l12), X) > 0 and np.dot(np.transpose(l24), X) > 0 and np.dot(np.transpose(l43),
                                                                                                    X) > 0 and np.dot(
                        np.transpose(l31), X) > 0:
                    if X_prime[1] < ho and X_prime[0] < wo:
                        x_prime_img[r][c] = x_img[X_prime[1]][X_prime[0]]
    # plt.imshow(x_prime_img)
    # plt.show()
    save_img('task2_r2d_card4.jpg', output, x_prime_img)

if __name__ == "__main__":
    main()