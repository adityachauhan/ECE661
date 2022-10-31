import os

import random
from scipy.optimize import least_squares

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter
from einops import rearrange
from tqdm import trange, tqdm
from scipy.signal import convolve2d
import sys
import BitVector
from sklearn import svm
from sklearn.metrics import confusion_matrix, \
    ConfusionMatrixDisplay, accuracy_score
from HW7Auxilliary.vgg import VGG19
from skimage import transform
def readImgCV(path):
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else: return None
    return img

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def bgr2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def cvrt2homo(pt):
    return np.append(pt, 1)

def gauss_blur(img, kernel_size=5):
    blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blur
def cv2show(img, name):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(img, size):
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img
def pltshow(img):
    plt.imshow(img)
    plt.show()
def plotCost(cost, name):
    plt.figure()
    plt.scatter(range(len(cost)), np.array(cost), s=80, edgecolors='black', c='red', label='data')
    plt.xlabel('Iterations')
    plt.xlabel('Cost')
    plt.title("Cost VS Iterations")
    plt.legend()
    plt.savefig(name)

def str2np(s):
    x_prime_pts = s.split(',')
    x_prime_pts = [int(val) for val in x_prime_pts]
    x_prime_pts = np.array(x_prime_pts)
    x_prime_pts = rearrange(x_prime_pts, '(c h)-> c h ', c=4, h=2)
    return x_prime_pts

def save_img(name, path, img):
    img_path = os.path.join(path, name)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path, img)

def save_img_v2(name, path, img):
    img_path = os.path.join(path, name)
    cv2.imwrite(img_path, img)
def xpeqhx(x,y,h):
    X = np.array((x, y, 1))
    X_prime = np.matmul(h, X)
    X_prime = X_prime / X_prime[2]
    X_prime = X_prime.astype(int)
    return X_prime

def point2point(x,h):
    X = np.array((x[0], x[1], 1))
    X_prime = np.matmul(h, X)
    X_prime = X_prime/X_prime[2]
    X_prime = X_prime.astype(int)
    X_prime = np.array([X_prime[0], X_prime[1]])
    return X_prime
def vanishing_line_homography(x_pts):
    l12 = np.cross(cvrt2homo(x_pts[0]), cvrt2homo(x_pts[1]))
    l34 = np.cross(cvrt2homo(x_pts[2]), cvrt2homo(x_pts[3]))
    l31 = np.cross(cvrt2homo(x_pts[2]), cvrt2homo(x_pts[0]))
    l42 = np.cross(cvrt2homo(x_pts[3]), cvrt2homo(x_pts[1]))
    vpt1 = np.cross(l12, l34)
    vpt1 = vpt1 / vpt1[2]
    vpt2 = np.cross(l31, l42)
    vpt2 = vpt2 / vpt2[2]
    vl = np.cross(vpt1, vpt2)
    h = np.zeros((3,3))
    h[0][0] = 1
    h[1][1] = 1
    h[2][0] = vl[0]/vl[2]
    h[2][1] = vl[1]/vl[2]
    h[2][2] = vl[2]/vl[2]
    return h, vl

def primePtsP2P(pts):
    X=np.ones((len(pts),2))
    # print(X)
    X[0,0]=pts[0,0]
    X[0,1]=pts[0,1]
    X[1,0]=pts[1,0]
    X[1,1]=pts[0,1]
    X[2,0]=pts[0,0]
    X[2,1]=pts[2,1]
    X[3,0]=pts[1,0]
    X[3,1]=pts[2,1]
    return X
def make_line_hc(pts1, pts2):
    l = np.cross(cvrt2homo(pts1), cvrt2homo(pts2))
    return l

class Vision:
    def __init__(self, x, x_prime, homo_mat_size=3):
        self.x = x
        self.x_prime = x_prime
        self.h = np.ones((homo_mat_size,homo_mat_size))
        self.h_mat_size = homo_mat_size

    def calc_homograpy(self, homo_mode):
        if homo_mode=='projective':
            A = np.ones((len(self.x)*2,len(self.x)*2))
            C = np.ones(len(self.x)*2)
            for i in range(len(self.x)):
                A[2*i] = np.array([self.x[i][0], self.x[i][1], 1, 0, 0, 0, -self.x[i][0]*self.x_prime[i][0], -self.x[i][1]*self.x_prime[i][0]])
                A[(2*i)+1] = np.array([0, 0, 0, self.x[i][0], self.x[i][1], 1, -self.x[i][0]*self.x_prime[i][1], -self.x[i][1]*self.x_prime[i][1]])
                C[2*i] = self.x_prime[i][0]
                C[(2*i)+1] = self.x_prime[i][1]
            Ainv = np.linalg.inv(A)
            B = np.dot(Ainv, C)
            B = np.append(B, 1)
            self.h = rearrange(B, '(c h)-> c h',c=self.h_mat_size, h=self.h_mat_size)
        if homo_mode=='affine':
            A = np.ones(((len(self.x)-1) * 2, (len(self.x) -1)* 2))
            C = np.ones((len(self.x) -1) * 2)
            for i in range(len(self.x)-1):
                A[2 * i] = np.array([self.x[i][0], self.x[i][1], 1, 0, 0, 0])
                A[(2 * i) + 1] = np.array([0, 0, 0, self.x[i][0], self.x[i][1], 1])
                C[2 * i] = self.x_prime[i][0]
                C[(2 * i) + 1] = self.x_prime[i][1]
            Ainv = np.linalg.inv(A)
            B = np.dot(Ainv, C)
            B = np.append(B, (0,0,1))
            self.h = rearrange(B, '(c h)-> c h', c=self.h_mat_size, h=self.h_mat_size)
        return self.h


def harrisCornerDetector(img, filter="HAAR", sigma=1.2):
    # if filter=="SOBEL":
    #     gx, gy = Sobel()
    #
    # if filter=="HAAR":
    gx, gy = Haar_Wavelet(sigma)
    win_size = int(np.ceil(5 * sigma)) if np.ceil(5 * sigma) % 2 == 0 else int(np.ceil(5 * sigma)) + 1
    dx = convolve2d(img, gx, mode="same")
    dy = convolve2d(img, gy, mode="same")

    sumK = np.ones((win_size,win_size))
    dx2 = dx**2
    dy2 = dy**2
    dxdy= np.multiply(dx,dy)

    sum_dx2=convolve2d(dx2, sumK,mode="same")
    sum_dy2=convolve2d(dy2, sumK,mode="same")
    sum_dxdy=convolve2d(dxdy, sumK,mode="same")

    detC = sum_dx2*sum_dy2 - (sum_dxdy**2)
    trC = sum_dx2+sum_dy2
    trC2 = trC**2
    k=0.05
    Ratios = detC - k*trC2
    Ratios*(Ratios==maximum_filter(Ratios, footprint=np.ones((5,5))))
    corners = np.where(Ratios>0.01*np.max(Ratios))

    # print(len(corners[0]), len(corners[1]))
    return corners

def extarct_patch(pt, win_size, img):
    # print(img.shape)
    assert win_size%2 != 0
    pad = int((win_size-1)/2)
    val_minx = pt[0] - pad
    val_maxx = pt[0] + pad
    val_miny = pt[1] - pad
    val_maxy = pt[1] + pad
    win_minx = val_minx #if val_minx > 0 else 0
    win_maxx = val_maxx #if val_maxx < img.shape[1] else img.shape[1]
    win_miny = val_miny #if val_miny > 0 else 0
    win_maxy = val_maxy #if val_maxy < img.shape[0] else img.shape[0]
    # print(win_minx, win_miny, win_maxx, win_maxy)
    pix_vals = []
    for i in range(win_minx, win_maxx):
        for j in range(win_miny, win_maxy):
            pix_vals.append(img[j,i])
    return pix_vals

def ssd(pts1, pts2, img1, img2, win_size):
    combined_pt_set=[]
    patches1=[]
    patches2=[]
    pad=int((win_size-1)/2)
    img1=cv2.copyMakeBorder(img1, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    img2=cv2.copyMakeBorder(img2, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    for pt in pts1:
        patches1.append(extarct_patch(pt, win_size, img1))
    for pt in pts2:
        patches2.append(extarct_patch(pt, win_size, img2))

    for i in range(len(patches1)):
        min_dist =[]
        for j in range(len(patches2)):
            min_dist.append(np.mean((np.array(patches1[i])-np.array(patches2[j]))**2))
        min_val = min(v for v in min_dist if v>0.0)
        # combined_pt_set.append([i,np.argmin(np.array(min_dist)), min_dist[np.argmin(np.array(min_dist))]])
        combined_pt_set.append([i,min_dist.index(min_val), min_dist[min_dist.index(min_val)]])

    return combined_pt_set


def ncc(pts1, pts2, img1, img2, win_size):
    combined_pt_set = []
    patches1 = []
    patches2 = []
    pad = int((win_size - 1) / 2)
    img1 = cv2.copyMakeBorder(img1, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    img2 = cv2.copyMakeBorder(img2, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    for pt in pts1:
        patches1.append(extarct_patch(pt, win_size, img1))
    for pt in pts2:
        patches2.append(extarct_patch(pt, win_size, img2))

    for i in range(len(patches1)):
        min_dist = []
        p1=np.array(patches1[i])
        for j in range(len(patches2)):
            p2 = np.array(patches2[j])
            numerator = np.sum((p1-np.mean(p1))*(p2-np.mean(p2)))
            denomiator= np.sqrt(np.sum((p1-np.mean(p1))**2)*np.sum((p2-np.mean(p2))**2))
            min_dist.append(1-(numerator/(denomiator+1e-6)))
        min_val = min(v for v in min_dist if v > 0.0)
        combined_pt_set.append([i,min_dist.index(min_val), min_dist[min_dist.index(min_val)]])

    return combined_pt_set

'''
Sift code from opencv documentation page.
https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
'''
def sift_matching(img_path1, img_path2):
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    kp1, dv1 = sift.detectAndCompute(img1, None)
    kp2, dv2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(dv1,dv2,k=2)

    points = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            points.append([m])

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, points, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img3, points
def Sobel():
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return gx, gy

def Haar_Wavelet(sigma):
    size = int(np.ceil(4 * sigma)) if np.ceil(4 * sigma) % 2 == 0 else int(np.ceil(4 * sigma)) + 1
    gx = np.ones((size, size))
    gx = gx.astype(int)
    gx[:, :int(size / 2)] = -1
    gy = gx.T
    return gx, gy


def pick_color():
    return (random.randint(10,250),random.randint(10,250),random.randint(10,250))

def SIFTpoints(img):
    pts_set=[]
    sift = cv2.SIFT_create()
    pts, des = sift.detectAndCompute(img, None)

    for pt in pts:
        pts_set.append([int(pt.pt[0]), int(pt.pt[1])])
    return pts_set, des


def flann_matching(pts1,pts2,des1,des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    matchingPts1=[]
    matchingPts2=[]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.25 * n.distance:
            matchingPts1.append((pts1[m.queryIdx].pt[0],pts1[m.queryIdx].pt[1]))
            matchingPts2.append((pts2[n.trainIdx].pt[0],pts2[n.trainIdx].pt[1]))
    matchingPts1 = [[int(val[0]), int(val[1])] for val in matchingPts1]
    matchingPts2 = [[int(val[0]), int(val[1])] for val in matchingPts2]
    return matchingPts1, matchingPts2

def hmat_pinv(x, x_prime):
    A = np.ones((len(x) * 2, 8))
    C = np.ones(len(x) * 2)
    for i in range(len(x)):
        A[2 * i] = np.array([x[i][0], x[i][1], 1, 0, 0, 0, -x[i][0] * x_prime[i][0],
                             -x[i][1] * x_prime[i][0]])
        A[(2 * i) + 1] = np.array([0, 0, 0, x[i][0], x[i][1], 1, -x[i][0] * x_prime[i][1],
                                   -x[i][1] * x_prime[i][1]])
        C[2 * i] = x_prime[i][0]
        C[(2 * i) + 1] = x_prime[i][1]

    h = (np.linalg.inv((A.T)@A)@A.T)@C
    h = np.concatenate([h,[1]])
    h = rearrange(h, '(c h)-> c h',c=3, h=3)
    return h

def RANSAC(pts1, pts2):
    p = 0.99
    n = 20
    sigma=4
    delta = 3*sigma
    e = 0.1
    N = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** n)))
    print(N)
    n_total = len(pts1)
    pts1_hc = np.ones((n_total, 3))
    pts1_hc[:, :-1] = pts1
    pts2_hc = np.ones((n_total, 3))
    pts2_hc[:, :-1] = pts2
    # print(pts1_hc.shape, pts2_hc.shape)
    outliers=True
    cost = []
    while outliers:
        count=0
        prev_size=0
        idx_set=[]
        while count < N:
            randIdx = np.random.randint(n_total, size=n)
            cpts1 = pts1_hc[randIdx,:]
            cpts2 = pts2_hc[randIdx,:]
            cH = hmat_pinv(cpts1, cpts2)
            est_cpts2 = cH @ pts1_hc.T
            est_cpts2 = est_cpts2/est_cpts2[2]
            est_cpts2 = rearrange(est_cpts2, 'c h -> h c')
            error = (pts2_hc[:,:-1] - est_cpts2[:,:-1])**2
            error = error @ np.ones((2,1))
            cost.append(np.linalg.norm(error))
            in_indices = np.where(error <= delta)[0]
            if len(in_indices) > prev_size:
                print("Processing RANSAC....")
                idx_set = in_indices
                if len(idx_set) > (1 - e) * n_total:
                    outliers = False
                    break
                prev_size=len(in_indices)
            count+=1
        if len(idx_set) > (1-e)*n_total:
            outliers=False
        else:
            e*=2
            N = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** n)))
    print("num inliers found:", len(idx_set))
    return idx_set, cost

#Euclidian distance based matching found the matcher code online
def sift_match(pts1, des1, pts2, des2, tr=3):
    match_pts1 = []
    match_pts2 = []
    eu = np.zeros((len(pts1), len(pts2)))
    for i in trange(len(pts1)):
        for j in range(len(pts2)):
            eu[i, j] = np.linalg.norm(des1[i, :] - des2[j, :])

    eu = eu / np.min(eu)
    dy_threshold = tr
    for i in trange(len(pts1)):
        eu_min = np.min(eu[i, :])
        if eu_min < dy_threshold:
            j_min = np.argmin(eu[i, :])
            match_pts1.append(pts1[i])
            match_pts2.append(pts2[j_min])
    print('number of sift matches found:', len(match_pts1))
    return match_pts1, match_pts2

def costFun(h, pts1, pts2):
    h = rearrange(h, '(c h)-> c h', c=3,h=3)
    n_total = len(pts1)
    pts1_hc = np.ones((n_total, 3))
    pts1_hc[:, :-1] = pts1
    pts2_hc = np.ones((n_total, 3))
    pts2_hc[:, :-1] = pts2
    est_cpts2 = h @ pts1_hc.T
    est_cpts2 = est_cpts2 / est_cpts2[2]
    est_cpts2 = rearrange(est_cpts2, 'c h -> h c')
    X = rearrange(np.array(pts2), 'c h -> (c h)')
    f = rearrange(np.array(est_cpts2[:,:-1]), 'c h -> (c h)')
    error = X-f
    return error

def create_panaroma(curr_img, new_img, H):
    hc,wc,cc = curr_img.shape
    hn,wn,cn = new_img.shape
    Hinv = np.linalg.inv(H)
    pts_hc = []
    for i in range(hc):
        for j in range(wc):
            pts_hc.append([j,i,1])
    pts_hc = np.array(pts_hc)
    est_pts = Hinv @ pts_hc.T
    est_pts = est_pts/est_pts[2,:]
    est_pts = est_pts.T[:,0:2].astype('int')
    idx= np.where(np.logical_and(np.logical_and(est_pts[:,0]>=0,est_pts[:,0]<=wn-1),np.logical_and(est_pts[:,1]>=0,est_pts[:,1]<=hn-1)))
    pts=pts_hc[idx]
    est_pts=est_pts[idx]
    curr_img[pts[:, 1], pts[:, 0]] = new_img[est_pts[:, 1], est_pts[:, 0]]
    return curr_img

def plot_inliers_outliers(img1, img2, idx, mp1, mp2):
    comb_img = np.concatenate((img1, img2), axis=1)
    for i in range(len(mp1)):
        p1 = (int(mp1[i][0]), int(mp1[i][1]))
        p2 = (int(mp2[i][0]) + img1.shape[1], int(mp2[i][1]))
        if i in idx:
            cv2.circle(comb_img, p1, radius=3, color=(0, 255, 0),thickness=-1)
            cv2.circle(comb_img, p2, radius=3, color=(0, 255, 0),thickness=-1)
            cv2.line(comb_img, p1, p2, color=(0, 255, 0), thickness=1)

        else:
            cv2.circle(comb_img, p1, radius=3, color=(255, 0, 0), thickness=-1)
            cv2.circle(comb_img, p2, radius=3, color=(255, 0, 0),thickness=-1)
            cv2.line(comb_img, p1, p2, color=(255, 0, 0), thickness=1)

    return comb_img

def Jacobian(pts, h):
    h = rearrange(h, '(c h) -> c h', c=3,h=3)
    J = np.zeros((len(pts)*2, 9))
    for i in range(len(pts)):
        f=h @ np.array([pts[i][0], pts[i][1], 1])
        J[2*i] = np.array([pts[i][0]/f[-1], pts[i][1]/f[-1], 1/f[-1], 0, 0, 0, (-pts[i][0]*f[0])/(f[-1]**2), (-pts[i][1]*f[0])/(f[-1]**2), -f[0]/(f[-1]**2)])
        J[2*i+1] = np.array([0,0,0,pts[i][0]/f[-1], pts[i][1]/f[-1], 1/f[-1], (-pts[i][0]*f[1])/(f[-1]**2), (-pts[i][1]*f[1])/(f[-1]**2), -f[1]/(f[-1]**2)])

    return J

#Followed from the pseudo-code by levmar
def LevMar(H, pts1, pts2):
    I = np.identity(9)
    tau = 0.5
    J = Jacobian(pts1, H)
    A = J.T @ J
    mu = tau * np.max(np.diag(A))
    h=H
    k=0
    kmax=100
    cost=[]
    while k<kmax:
        ce = costFun(h, pts1, pts2)
        c = np.linalg.norm(ce)**2
        Jk = Jacobian(pts1, h)
        deltaP = (np.linalg.inv((Jk.T @ Jk) + mu*I) @ Jk.T) @ ce
        Hk=h + deltaP
        cek = costFun(Hk, pts1, pts2)
        ck = np.linalg.norm(cek)**2
        cost.append(ck)
        rho_num = c-ck
        rho_den = deltaP.T@(((mu*I) @ deltaP) + (Jk.T @ ce))
        rho = rho_num/rho_den
        muk = mu * max(1/3, 1-(2*rho - 1)**3)
        mu = muk
        h=Hk
        k+=1

    return h, cost

def findmaxmin(ho,wo,h):
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
    return wpa, hpa

def plthist(hist, bins):
    plt.bar(bins, hist, color='b', width=5, align='center', alpha=0.25)
    plt.show()
def histogram(img, bins):
    h,w = img.shape[0],img.shape[1]
    hist = np.zeros(bins)
    for i in range(h):
        for j in range(w):
            hist[img[i,j]]+=1
    return hist

def otsu(img, bins, hist):
    p = hist/np.sum(hist)
    ip=np.arange(1,bins+1)
    mu = np.multiply(p,ip)
    mut= p@ip
    sigmab=np.zeros(bins+1)
    for k in range(1, bins):
        w0 = np.sum(p[:k])
        w1 = 1-w0
        if w0 > 0 and w0 < 1 and w1 > 0 and w1 < 1:
            mu0 = np.sum(mu[:k])
            mu1 = (mut-mu0)/w1
            sigmab[k]=(w0*w1*((mu1-mu0)**2))
    ks = np.argmax(sigmab)
    print(ks)
    _,thresh = cv2.threshold(img, ks, 255, cv2.THRESH_BINARY)
    _,thresh_inv = cv2.threshold(img, ks, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh, thresh_inv



def otsu_rgb(img, bins):
    channels = cv2.split(img)
    chcount = len(channels) if len(channels)==3 else 3
    comb = np.zeros(np.append(channels[0].shape, chcount))*255
    combinv = np.zeros(np.append(channels[0].shape, chcount))*255
    for i in range(len(channels)):
        hist = histogram(channels[i], bins)
        thresh, thresh_inv = otsu(channels[i], bins, hist)
        combinv[:,:,i] = thresh_inv
        comb[:,:,i] = thresh
        # comb = cv2.bitwise_and(comb, thresh_inv)
    return comb, combinv

def channel_and(channels):
    img = cv2.bitwise_and(cv2.bitwise_and(channels[:,:,0], channels[:,:,1]), channels[:,:,2])
    return img

def channel_and_v2(c1,c2,c3):
    img = cv2.bitwise_and(cv2.bitwise_and(c1,c2),c3)
    return img


def otsu_texture(img, window_sizes):
    comb = np.zeros(np.append(img.shape, len(window_sizes)))
    for idx in range(len(window_sizes)):
        padding = window_sizes[idx]//2
        temp = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                win = temp[r:r+(2*padding+1),c:c+(2*padding+1)]
                comb[r,c,idx] = np.var(win-np.mean(win))
    comb=comb.astype(np.uint8)
    return comb
def contour(img):
    img = img//255
    cnt = np.zeros(img.shape).astype(np.uint8)
    padding = 1
    temp = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    for r in range(padding,img.shape[0]):
        for c in range(padding,img.shape[1]):
            if temp[r,c]==0: continue
            if np.sum(img[r-padding:r+padding+1, c-padding:c+padding+1])<9:
                cnt[r,c]=255
    return cnt


def dilate(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    return img

def erode(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    return img

def opening(img, kd, ke, num):
    for _ in range(num):
        img = dilate(img, kd)
        img = erode(img, ke)
    return img

def closing(img, kd, ke, num):
    for _ in range(num):
        img = erode(img, ke)
        img = dilate(img, kd)
    return img

def make_border(img, padding, value=0):
    img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=value)
    return img
def LBP(img, size=(64,64), R=1, P=8):
    h,w=img.shape[0], img.shape[1]
    gray = bgr2gray(img)
    gray = resize(gray, size)
    # cv2show(gray, 'gray')
    padding=1
    gray = make_border(gray, padding)
    hist = [0 for _ in range(P+1)]
    for i in range(padding, gray.shape[0]-1):
        for j in range(padding, gray.shape[1]-1):
            win = gray[i-padding:i+padding+1, j-padding:j+padding+1]
            intensities = interpolate(win,R,P)
            # print(intensities)
            pattern = np.where(intensities >= gray[i,j],1,0)
            # print(pattern)
            enc = encode(pattern, P)
            # print(enc)
            hist[enc-1]+=1

    return hist



def interpolate(win,R=1,P=8):
    X = np.cos(np.array([(np.pi / (P / 2)) * p for p in range(P)])) * R
    Y = np.sin(np.array([(np.pi / (P / 2)) * p for p in range(P)])) * R
    vals = [0 for _ in range(P)]
    # print(X)
    # print(Y)
    for i in range(1,len(X),2):
        p1 = np.array((int(np.floor(1+X[i])), int(np.floor(1+Y[i]))))
        p2 = np.array((int(np.ceil(1+X[i])), int(np.floor(1+Y[i]))))
        p3 = np.array((int(np.floor(1+X[i])), int(np.ceil(1+Y[i]))))
        p4 = np.array((int(np.ceil(1+X[i])), int(np.ceil(1+Y[i]))))
        # print("points",p1,p2,p3,p4)
        p = np.array((X[i], Y[i]))
        
        d1 = 1/np.linalg.norm(p1-p)
        d2 = 1/np.linalg.norm(p2-p)
        d3 = 1/np.linalg.norm(p3-p)
        d4 = 1/np.linalg.norm(p4-p)
        # print("distances",d1,d2,d3,d4)
        inten = d1*win[p1[0],p1[1]] + d2*win[p2[0],p2[1]] + d3*win[p3[0],p3[1]] + d4*win[p4[0],p4[1]]
        inten/=(d1+d2+d3+d4+1e-16)
        vals[i] = inten
    vals[0] = win[2,1]
    vals[2] = win[1,2]
    vals[4] = win[0,1]
    vals[6] = win[1,0]
    return np.array(vals)

def encode(pattern,P):
    # print(pattern)
    pattern = pattern.tolist()
    bv = BitVector.BitVector(bitlist=pattern)
    val = [int(bv << 1) for _ in range(P)]
    minbv = BitVector.BitVector(intVal = min(val), size=P)
    runs = minbv.runs()
    if len(runs)>2:
        enc = P+1
    elif len(runs) == 1 and runs[0][0]=='1':
        enc = P
    elif len(runs) == 1 and runs[0][0]=='0':
        enc = 0
    else:
        enc = len(runs[1])
    return enc


def getLabel(name):
    label=0
    if "cloudy" in name : label = 0
    if "rain" in name : label = 1
    if "shine" in name : label = 2
    if "sunrise" in name : label = 3
    return label

def train(X_train,Y_train, X_test):
    clf = svm.SVC()
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    return Y_pred

def display_conf_mat(conf_mat, labels, path, acc_str):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
    disp.plot()
    plt.title("Accuracy :" + acc_str)
    plt.savefig(path)
    plt.show()
def evaluate(y_true, y_pred, path, labels):
    acc_score = accuracy_score(y_true, y_pred)
    conf_matx = confusion_matrix(y_true, y_pred)
    print("Model Accuracy: ", acc_score)
    acc_str = str(acc_score*100) + " %"
    print("Confusion Matrix: ", conf_matx)
    # labels = [i for i in range(4)]
    display_conf_mat(conf_matx, labels, path, acc_str)

def classify(X_train, Y_train, X_test, Y_test, path, labels):
    pred_labels = train(X_train, Y_train, X_test)
    evaluate(Y_test, pred_labels, path, labels)

def vgg_feature_extractor(img, model_path, size=(256,256)):
    vgg = VGG19()
    vgg.load_weights(model_path)
    img = transform.resize(img, size)
    feature = vgg(img)
    return feature

def channelNorm(features):
    features = rearrange(features, 'c h w -> c (h w)')
    mu = np.mean(features, axis=1)
    var = np.var(features, axis=1)
    texture_des = np.zeros(mu.shape[0]+var.shape[0], dtype=mu.dtype)
    texture_des[::2] = mu
    texture_des[1::2] = var
    return texture_des



