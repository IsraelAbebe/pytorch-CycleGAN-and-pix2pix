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

def save_combined(im, path, filename):

    wsize = 512  # double the resolution 1024
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

    print('concat image saved')

def sketch(im, color_pic, filename):
    Gamma = 0.97 #0.97
    Phi = 200
    Epsilon = 0.5 #0.5
    k = 2
    Sigma = 1.5

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

            return gray_pic, org_pic

def save_gen(gen, sketch, filename):
    print("GENERATED: ", gen)
    print("FILENAME: ", filename)
    print("ONLY FILENAME:", os.path.basename(filename))
    sketch.save(os.path.join(gen, 't' + filename))
    print('gray image', os.path.join(gen, 't' + filename), " saved")
    return sketch

def save_orgtogen(gray_pic, org_pic, orgtogen, filename):

    combined_pic = np.append(org_pic, gray_pic, axis=1)
    concat_img = Image.fromarray(combined_pic)
    save_combined(concat_img, orgtogen, filename)
    return concat_img

def save_gentoorg(gray_pic, org_pic, gentoorg, filename):

    combined_pic = np.append(gray_pic, org_pic, axis=1)
    concat_img = Image.fromarray(combined_pic)
    save_combined(concat_img, gentoorg, filename)
    return concat_img

def save_results(im, color_pic, filename, gen, orgtogen, gentoorg):

    gray_pic, org_pic = sketch(im, color_pic, filename)
    sketch_pic = Image.fromarray(gray_pic, mode = 'RGB')
    print(filename)
    if gen:
        if not os.path.exists(gen): os.mkdir(gen)
        save_gen(gen, sketch_pic, os.path.basename(filename))
    if orgtogen:
        if not os.path.exists(orgtogen): os.mkdir(orgtogen)
        save_orgtogen(gray_pic, org_pic, orgtogen, os.path.basename(filename))
    if gentoorg:
        if not os.path.exists(gentoorg): os.mkdir(gentoorg)
        save_gentoorg(gray_pic, org_pic, gentoorg, os.path.basename(filename))

def main(args):

    # args values
    input_dir = args.input_dir
    gen = args.gen
    orgtogen = args.orgtogen
    gentoorg = args.gentoorg
    input_image = args.input_image

    #parameter
    max_length=20
    min_length=10
    max_dif=30
    n_point=50
    dir = 3

    if input_image:
        #filepath, filename = os.path.split(files1)
        if not os.path.exists(input_image): os.mkdir(input_image)
        filename = input_image
        print("FILENAME:", filename)
        im = Image.open(filename).convert('L')
        color_pic = Image.open(filename)

        save_results(im, color_pic, filename, gen, orgtogen, gentoorg)

    if input_dir:
        input_paths = glob.glob(input_dir+ '/*.jpg')
        input_paths+=(glob.glob(input_dir+ '/*.jpeg'))
        input_paths+=(glob.glob(input_dir + '/*.png'))

        for files1 in input_paths:
            filepath, filename = os.path.split(files1)
            im = Image.open(files1).convert('L')
            color_pic = Image.open(files1)

            save_results(im, color_pic, filename, gen, orgtogen, gentoorg)

    if not input_dir and not input_image:
         print(parser.print_help(sys.stderr))
         sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--input_image', type=str)
    parser.add_argument('--gen', type=str)#, default="output")#, action="store_true")
    parser.add_argument('--orgtogen', type=str)
    parser.add_argument('--gentoorg', type=str)#, nargs='?')
    args = parser.parse_args()
    main(args)
