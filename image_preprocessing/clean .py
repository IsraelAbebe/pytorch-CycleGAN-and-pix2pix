import cv2
import numpy as np

import os
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import filters
class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

def remove_dots(img):
    #img = cv2.imread(file1, 0)
    _, blackAndWhite = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 50:   #filter small dotted regions
            img2[labels == i + 1] = 255

    res = cv2.bitwise_not(img2)
    result = cv2.imwrite(file1, res)
    return result


def merge_images(image1, image2):
    #image1 = Image.open(file1)
    #image2 = Image.open(file2)
    img = np.concatenate((image1, image2), axis=1)
    return img
    """(width1, height1) = image1.size
    (width2, height2) = image2.size

    # result_width = width1 + width2
    result_width = width1 *2
    # result_height = max(height1, height2)

    result_height = height1
    print (height2)
    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(height1,0))
    result = result.resize((512,256), Image.ANTIALIAS)
    result.save("result/"+file1)"""


def image_resize(img):
    #print("saving "+file1)
    #img = Image.open(file1)
    #img = Image.fromarray(img)
    #img = img.resize((512,256), Image.ANTIALIAS)
    img = cv2.resize(img, (400,256), interpolation = cv2.INTER_AREA)
    #img.save(file1)
    return img


def image_grayscale(img):
    #img = cv2.imread(file1, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #fname, ext = os.path.splitext(file1)
    #result = cv2.imwrite(fname+"1."+ext, gray)
    return gray


def facecrop(img):
    facedata = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)

    #img = cv2.imread(image)
    
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)
    face_cont = []
    for f in faces:
        x, y, w, h = [ v for v in f ]
        # cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y-100:y+h+100, x-100:x+w+100]
        #fname, ext = os.path.splitext(image)
        #cv2.imwrite(fname+"_cropped_"+ext, sub_face)
        face_cont.append(sub_face)
    return face_cont[0]
def sketch(im, color_pic, filename):
    Gamma = 0.97 #0.97
    Phi = 200
    Epsilon = 0.5 #0.5
    k = 2
    Sigma = 1.5
    im = Image.fromarray(im)
    color_pic = Image.fromarray(color_pic)
    im = np.array(ImageEnhance.Sharpness(im).enhance(5.0)) #3 neber
    im2 = filters.gaussian_filter(im, Sigma)
    im3 = filters.gaussian_filter(im, Sigma * k)
    differencedIm2 = im2 - (Gamma * im3)
    (x, y) = np.shape(im2)
    for i in range(x):
        for j in range(y):
            if differencedIm2[i, j] < Epsilon:
                differencedIm2[i, j] = 1
            else:
                differencedIm2[i, j] = 250 + np.tanh(Phi * (differencedIm2[i, j]))

    gray_pic = differencedIm2.astype(np.uint8)

    org_pic = np.atleast_2d(color_pic)

    if org_pic.ndim == 2:
        org_pic = np.stack((org_pic, org_pic, org_pic),axis=2)

    if org_pic.ndim == 3:
        w, h, c = org_pic.shape
        if c>0:
            image = color_pic.filter(MyGaussianBlur(radius=5))
            mat = np.atleast_2d(image)

            if gray_pic.ndim == 2:
                gray_pic = np.expand_dims(gray_pic, 2)
                gray_pic = np.tile(gray_pic, [1, 1, c]) # last one 3
            
            return np.array(gray_pic), np.array(org_pic)


# sketch_dir = "result/face_emotions/"
# images = os.listdir(sketch_dir)
# remove_dots("t00002.jpg")
# merge_images("00002.jpg","t00002.jpg")
# image_resize("result/00002.jpg")

# source_dir = "output/"
# dir = os.listdir(source_dir)
# for i in dir:
#     res = remove_dots(source_dir+i)
#     print(res)
