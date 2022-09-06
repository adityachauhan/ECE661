import cv2
import numpy as np
import matplotlib.pyplot as plt

img_car = cv2.imread('../hw2images/car.jpg')
card1 = cv2.imread('../hw2images/card2.jpeg')
card1 = cv2.cvtColor(card1, cv2.COLOR_BGR2RGB)
img_car = cv2.cvtColor(img_car, cv2.COLOR_BGR2RGB)
h,w,c = img_car.shape
hp,wp,cp=card1.shape
print(h,w)
print(hp,wp)
# cv2.imshow('img',img_car)
# cv2.waitKey(0)
# plt.imshow(card1)
# plt.show()
#card1
# cp1 = (525,287)
# cp2 = (1225,209)
# cp3 = (633,1080)
# cp4 = (1207,792)
#card2
cp1 = (330,259)
cp2 = (1008,266)
cp3 = (226,849)
cp4 = (856,1092)
#card3
# cp1 = (588,87)
# cp2 = (1194,683)
# cp3 = (92,597)
# cp4 = (700,1162)

card1_pts = np.array([cp1, cp2, cp3, cp4])
# print(card1_pts)

carp1=(0,0)
carp2=(w,0)
carp3=(0,h)
carp4=(w,h)
car_pts = np.array([carp1, carp2, carp3, carp4])
# print(car_pts)

A1 = np.array([car_pts[0][0], car_pts[0][1], 1, 0,0,0,-car_pts[0][0]*card1_pts[0][0],-car_pts[0][1]*card1_pts[0][0]])
A2 = np.array([0,0,0,car_pts[0][0], car_pts[0][1], 1,-car_pts[0][0]*card1_pts[0][1],-car_pts[0][1]*card1_pts[0][1]])
A3 = np.array([car_pts[1][0], car_pts[1][1], 1, 0,0,0,-car_pts[1][0]*card1_pts[1][0],-car_pts[1][1]*card1_pts[1][0]])
A4 = np.array([0,0,0,car_pts[1][0], car_pts[1][1], 1,-car_pts[1][0]*card1_pts[1][1],-car_pts[1][1]*card1_pts[1][1]])
A5 = np.array([car_pts[2][0], car_pts[2][1], 1, 0,0,0,-car_pts[2][0]*card1_pts[2][0],-car_pts[2][1]*card1_pts[0][0]])
A6 = np.array([0,0,0,car_pts[2][0], car_pts[2][1], 1,-car_pts[2][0]*card1_pts[2][1],-car_pts[2][1]*card1_pts[2][1]])
A7 = np.array([car_pts[3][0], car_pts[3][1], 1, 0,0,0,-car_pts[3][0]*card1_pts[3][0],-car_pts[3][1]*card1_pts[3][0]])
A8 = np.array([0,0,0,car_pts[3][0], car_pts[3][1], 1,-car_pts[3][0]*card1_pts[3][1],-car_pts[3][1]*card1_pts[3][1]])

A = np.array([A1,A2,A3,A4,A5,A6,A7,A8])
C = np.array([card1_pts[0][0],card1_pts[0][1],card1_pts[1][0],card1_pts[1][1],card1_pts[2][0],card1_pts[2][1],card1_pts[3][0],card1_pts[3][1]])
Ainv = np.linalg.inv(A)
B = np.dot(Ainv, C)
print(B)
H = np.array([[B[0],B[1], B[2]],[B[3], B[4], B[5]],[B[6], B[7], 1]])
print(H)
for r in range(w):
    for c in range(h):
        X = np.array((r,c,1))
        X_prime = np.matmul(H, X)
        # print(X_prime.shape)
        X_prime = X_prime/X_prime[2]
        X_prime = X_prime.astype(np.int)
        # print(X_prime)
        # print(X_prime.shape)
        card1[X_prime[1]][X_prime[0]] = img_car[c][r]

plt.imshow(card1)
plt.show()