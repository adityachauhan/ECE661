import configparser
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from  utility import *
from vision import *

config = configparser.ConfigParser()
config.read('hw3config.txt')

def main():
    x_img_path = os.path.join(config['PARAMETERS']['top_dir'], config['PARAMETERS']['x_path'])
    x_pts = config['PARAMETERS']['ol_pts1']
    homo_mode = config['PARAMETERS']['homo_mode']
    x_img = readImgCV(x_img_path)

    ho, wo, co = x_img.shape
    print(ho,wo,co)

    #############################################################################################
    ##################     Homography calculations for P2P correspondenc       ##################
    #############################################################################################

    x_pts = str2np(x_pts)
    x_prime_pts=primePtsP2P(x_pts)
    vision = Vision(x_pts, x_prime_pts)
    h = vision.calc_homograpy(homo_mode)
    hinv=np.linalg.inv(h)


    hPrime = xpeqhx(0, 0, h)
    wpa1, hpa1 = hPrime[0], hPrime[1]
    hPrime = xpeqhx(0, ho, h)
    wpa2, hpa2 = hPrime[0], hPrime[1]
    hPrime = xpeqhx(wo, 0, h)
    wpa3, hpa3 = hPrime[0], hPrime[1]
    hPrime = xpeqhx(wo, ho, h)
    wpa4, hpa4 = hPrime[0], hPrime[1]
    wpa = max(wpa1, max(wpa2, max(wpa3, wpa4)))
    hpa = max(hpa1, max(hpa2, max(hpa3, hpa4)))

    #############################################################################################
    ##################     Loop over empty image to fetch the coord from the images       #######
    #############################################################################################
    empty_img = np.ones((int(hpa), int(wpa), 3), np.int32)
    # print(empty_img.shape)
    for c in tqdm(range(int(wpa))):
        for r in range(int(hpa)):
            X_prime=xpeqhx(c,r,hinv)
            X_prime=X_prime.astype(int)
            if X_prime[1] < ho and X_prime[0] < wo and X_prime[0]>=0 and X_prime[1]>=0:
                empty_img[r][c] = x_img[X_prime[1]][X_prime[0]]
    plt.imshow(empty_img)
    plt.show()

if __name__=='__main__':
    main()