import cv2
img1 = cv2.imread('./samples/data/lena.jpg', -1)

cv2.imshow('image', img1)
while True:
    k = cv2.waitKey(0)
    if k == 27:
        print("K==27")
