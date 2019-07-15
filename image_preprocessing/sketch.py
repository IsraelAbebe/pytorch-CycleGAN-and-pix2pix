from PIL import Image, ImageEnhance, ImageFilter
from pylab import *
from scipy.ndimage import filters

import glob, os
import sys, getopt
import argparse


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


def start(mat):
    w, h, c = mat.shape
    ws = int(w * random(1))
    hs = int(h * random(1))
    while (mat[ws, hs, 0] > 254) and (mat[ws, hs, 1] > 254) and (mat[ws, hs, 2] > 254):
        ws = int(w * random(1))
        hs = int(h * random(1))
    return ws, hs


def gen_color_line(mat, dir,max_length,max_dif):
    w, h, c = mat.shape
    ws = int(w * random(1))
    hs = int(h * random(1))
    while (mat[ws, hs, 0] > 254) and (mat[ws, hs, 1] > 254) and (mat[ws, hs, 2] > 254):
        ws = int(w * random(1))
        hs = int(h * random(1))
    if dir == 1:
        wt = ws
        ht = hs
        while (wt < w - 1) and (abs(int(mat[wt, ht, 1]) + int(mat[wt, ht, 2]) + int(mat[wt, ht, 0]) - (
                        int(mat[wt + 1, ht, 1]) + int(mat[wt + 1, ht, 2]) + int(mat[wt + 1, ht, 0]))) < 80):
            wt = wt + 1
    if dir == 2:
        wt = ws
        ht = hs
        while (ht < h - 1) and (abs(int(mat[wt, ht, 1]) + int(mat[wt, ht, 2]) + int(mat[wt, ht, 0]) - (
                        int(mat[wt, ht + 1, 1]) + int(mat[wt, ht + 1, 2]) + int(mat[wt, ht + 1, 0]))) < 3):
            ht = ht + 1
    if dir == 3:
        wt = ws
        ht = hs
        length = 0
        while (length < max_length) and (wt < w-1) and (ht < h-1) and (
            abs(int(mat[wt, ht, 1]) + int(mat[wt, ht, 2]) + int(mat[wt, ht, 0]) - (
                            int(mat[wt + 1, ht + 1, 1]) + int(mat[wt + 1, ht + 1, 2]) + int(
                        mat[wt + 1, ht + 1, 0]))) < max_dif):
            ht += 1
            wt += 1
            length = abs(wt - ws) + abs(ht - hs)
    return ws, hs, wt, ht, length

def save_combined(img_arr, path, filename):

    wsize = 512  # double the resolution 1024
    train_count = 0
    final_img = Image.fromarray(img_arr)

    im = final_img
    w, h = im.size
    hsize = int(h * wsize / float(w))
    if hsize * 2 > wsize:  # crop to three
        im = im.resize((wsize, hsize))
        bounds1 = (0, 0, wsize, int(wsize / 2)) #/2
        cropImg1 = im.crop(bounds1)
        # cropImg1.show()
        cropImg1.save(os.path.join(path, 'u' + filename))
        bounds2 = (0, hsize - int(wsize / 2), wsize, hsize) #wsize/2

    else:
        im = im.resize((wsize // 2, (wsize // 4)))
        im.save(os.path.join(path, 't' + filename))#t

    train_count += 1
    print('train' + str(train_count))

def main(args):

    ifile = args.input
    gen = args.gen
    orgtogen = args.orgtogen
    gentoorg = args.gentoorg

    print("GEN::", gen, ".", orgtogen, ".", gentoorg)

    if not os.path.exists(gen): os.mkdir(gen)

    gray_count = 0
    #parameter

    Gamma = 0.97 #0.97
    Phi = 200
    Epsilon = 0.5 #0.5
    k = 2
    Sigma = 1.5
    max_length=20
    min_length=10
    max_dif=30
    n_point=50
    dir = 3

    if not ifile:
         print(parser.print_help(sys.stderr))
         sys.exit()

    input_paths = glob.glob(ifile+ '/*.jpg')
    input_paths+=(glob.glob(ifile+ '/*.jpeg'))
    input_paths+=(glob.glob(ifile + '/*.png'))

    for files1 in input_paths:
        filepath, filename = os.path.split(files1)

        im = Image.open(files1).convert('L')
        im_arr = np.array(im)

        #if im_arr.ndim == 2:
        #    img = np.stack((im_arr,im_arr,im_arr),axis=2)
        #    print("LEWET SHAPE", img.shape)
        #    im = Image.fromarray(img)


        im = array(ImageEnhance.Sharpness(im).enhance(5.0)) #3 neber
        im2 = filters.gaussian_filter(im, Sigma)
        im3 = filters.gaussian_filter(im, Sigma * k)
        differencedIm2 = im2 - (Gamma * im3)
        (x, y) = shape(im2)
        for i in range(x):
            for j in range(y):
                if differencedIm2[i, j] < Epsilon:
                    differencedIm2[i, j] = 1
                else:
                    differencedIm2[i, j] = 250 + tanh(Phi * (differencedIm2[i, j]))

        gray_pic = differencedIm2.astype(np.uint8)
        color_pic = Image.open(files1)
        real = np.atleast_2d(color_pic)

        if real.ndim == 2:
            real = np.stack((real, real, real),axis=2)

        if real.ndim == 3:
            w, h, c = real.shape
            if c>0:
                image = color_pic.filter(MyGaussianBlur(radius=5))
                mat = np.atleast_2d(image)

                if gray_pic.ndim == 2:
                    gray_pic = np.expand_dims(gray_pic, 2)
                    gray_pic = np.tile(gray_pic, [1, 1, c]) # last one 3

                sketch = Image.fromarray(gray_pic, mode = 'RGB')
                sketch.save(os.path.join(gen, 't' + filename))
                gray_count += 1
                print('gray' + str(gray_count))

                if args.orgtogen:# is not None:
                    if not os.path.exists(orgtogen): os.mkdir(orgtogen)
                    combined_pic = np.append(real, gray_pic, axis=1)
                    save_combined(combined_pic, orgtogen, filename)

                if args.gentoorg:
                    if not os.path.exists(gentoorg): os.mkdir(gentoorg)
                    combined_pic = np.append(gray_pic, real, axis=1)
                    save_combined(combined_pic, gentoorg, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--gen', type=str)#, default="output")#, action="store_true")
    parser.add_argument('--orgtogen', type=str)
    parser.add_argument('--gentoorg', type=str)#, nargs='?')
    args = parser.parse_args()
    main(args)
