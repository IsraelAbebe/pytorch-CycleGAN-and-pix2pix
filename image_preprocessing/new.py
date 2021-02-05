import cv2
import numpy as np

import os
import clean as cl
fname = 'mn.jpg'
img = cv2.imread(fname, 1)
cv2.imshow("Original", img)
cv2.waitKey(0)

image = cl.image_grayscale(img)
image = cl.facecrop(image)
#image = cl.image_resize(image)

image1,image2 = cl.sketch(image,image,fname)
cv2.imshow("Original", image1)
cv2.waitKey(0)
cv2.imwrite('new.jpg',image1)
image1= cl.remove_dots('new.jpg',50)
#image = cl.merge_images(image1,image2)
cv2.imshow("Original", image1)
cv2.waitKey(0)

