"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path
import cv2
from image_preprocessing.clean import remove_dots,merge_images,image_resize,image_grayscale,facecrop,sketch
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    if os.path.isdir(dir):
        if not os.path.exists('sketch_images/'):
            os.makedirs('sketch_images/')
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(dir, fname)
                    img = cv2.imread(path, 1)
                    img = image_grayscale(img)
                    print(fname)
                    faces = facecrop(img)
                    num = 0
                    for i in faces:
                        image1,image2 = sketch(i,img)
                        cv2.imwrite('sketch_images/'+fname,image1)
                        image1= remove_dots('sketch_images/'+fname,50)
                        cv2.imwrite('sketch_images/'+fname,image1) 
    else:
        if is_image_file(dir):
            dir1 = os.path.dirname(dir)
            fname = os.path.basename(dir)
            img = cv2.imread(dir, 1)
            img = image_grayscale(img)
            image = facecrop(img)
            image1,image2 = sketch(image,image)
            cv2.imwrite('sketch_images/'+fname,image1)
            image1= remove_dots('sketch_images/'+fname,50)
            cv2.imwrite('sketch_images/'+fname,image1)
    dir = 'sketch_images/'      
    images = []
    if os.path.isdir(dir):
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    else:
        images.append(dir)
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
