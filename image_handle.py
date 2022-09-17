# In[0]:
import cv2
import numpy as np
# In[1]:
# Deal with image
img1 = cv2.imread('Resources/Faces/train/Ben Afflek/1.jpg', -1)
cv2.imshow('image', img1)
while True:
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()
# In[2]:
# Rescale Frame


def rescaleFrame(frame, scale=.75):
    height = frame.shape[0]
    width = frame.shape[1]
    return cv2.resize(frame, (width, height),
                      interpolation=cv2.INTER_AREA)
# In[3]:
# Capture video from file
capture = cv2.VideoCapture("./Resources/Videos/dog.mp4")
# This function can take device no to capture video directly from camera device
try:
    isTrue = True
    while isTrue:
        isTrue, frame = capture.read()
        frame = rescaleFrame(frame, scale=.75)
        cv2.imshow('Video', frame)
        if cv2.waitKey(20) & 0xFF == ord('d'):
            break
except:
    pass
capture.release()

# In[4]:


# Read in an image
img = cv2.imread('./Resources/Photos/park.jpg')
cv2.imshow('Park', img)

# Converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

# Blur
blur = cv2.GaussianBlur(img, (7, 7), cv2.BORDER_DEFAULT)
cv2.imshow('Blur', blur)

# Edge Cascade
canny = cv2.Canny(blur, 125, 175)
cv2.imshow('Canny Edges', canny)

# Dilating the image
dilated = cv2.dilate(canny, (7, 7), iterations=3)
cv2.imshow('Dilated', dilated)

# Eroding
eroded = cv2.erode(dilated, (7, 7), iterations=3)
cv2.imshow('Eroded', eroded)

# Resize
resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
cv2.imshow('Resized', resized)

# Cropping
cropped = img[50:200, 200:400]
cv2.imshow('Cropped', cropped)


while cv2.waitKey(0) != ord('j'):
    pass

cv2.destroyAllWindows()

# In[5]:

img = cv2.imread('./Resources/Photos/cats.jpg')
cv2.imshow('Cats', img)

blank = np.zeros(img.shape, dtype='uint8')
cv2.imshow('Blank', blank)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
cv2.imshow('Blur', blur)

canny = cv2.Canny(blur, 125, 175)
cv2.imshow('Canny Edges', canny)

# ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
# cv2.imshow('Thresh', thresh)

contours, hierarchies = cv2.findContours(
    canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

cv2.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv2.imshow('Contours Drawn', blank)

while cv2.waitKey(0) != ord('j'):
    pass

cv2.destroyAllWindows()

# In[6]:


img = cv2.imread('./Resources/Photos/park.jpg')
cv2.imshow('Park', img)

# plt.imshow(img)
# plt.show()

# BGR to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

# BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV', hsv)

# BGR to L*a*b
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow('LAB', lab)

# BGR to RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('RGB', rgb)

# HSV to BGR
lab_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
cv2.imshow('LAB --> BGR', lab_bgr)

while cv2.waitKey(0) != ord('j'):
    pass

cv2.destroyAllWindows()
# In[7]:
# Bitwise Operator 

blank = np.zeros((400,400), dtype= 'uint8')
rectangle = cv2.rectangle


