import os
import LION.CTtools.ct_utils as ct
import LION.CTtools.ct_geometry as ctgeo
import numpy as np
import torch
from tomosipo.torch_support import (
    to_autograd,
)
import cv2 as cv



def is_image_file(filename):
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.TIF']
    return any(filename.endswith(extension) for extension in extensions)

def make_dataset(dir):
    img_paths = []
    assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)

    for (root, dirs, files) in sorted(os.walk(dir)):
        for filename in files:
            if is_image_file(filename):
                img_paths.append(os.path.join(root, filename))
    return img_paths




''' sobolev norm for loss in data domain, maybe it preserves details a bit better'''
def sobolev_norm(f,g):
     l2 = torch.nn.functional.mse_loss(f,g)
     im_size = f.shape[-2]
     ### f and g have shape (batch size, 1, N_s, N_theta)
     derivatives_s1 = torch.gradient(f, axis=(2,3))[0]
     derivatives_s2 = torch.gradient(g, axis=(2,3))[0]
     l2_grad = torch.nn.functional.mse_loss(derivatives_s1, derivatives_s2)
     #print("gradient term is: " + str(l2_grad))
     #print("l2 term is: " + str(l2))     
     sobolev = (l2**2 + l2_grad**2)/im_size**2
     return sobolev
# %%

photon_count=100   # 
attenuation_factor=2.76 # corresponds to absorption of 50%
def apply_noise(img, photon_count):
    opt = dict(dtype=np.float32)
    img = np.exp(-img, **opt)
    # Add poisson noise and retain scale by dividing by photon_count
    img = np.random.poisson(img * photon_count)
    img[img == 0] = 1
    img = img / photon_count
    # Redo log transform and scale img to range [0, img_max] +- some noise.
    img = -np.log(img, **opt)
    return img



def normalize_scan(image):
    norm_image = cv.normalize(
        image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F  # type: ignore
    )
    return norm_image


def resize_scan(scan, desired_width, desired_height):
    scan = cv.resize(scan, (desired_height, desired_width))
    return scan


def preprocess(clean, width, height):
    scan =np.copy(clean)
    resized_scan = resize_scan(scan, width, height)
    normalized_resized_scan = normalize_scan(resized_scan)
    return normalized_resized_scan

''' this function adds gaussian noise to our sinograms '''
def add_gaussian_noise(img, sigma):
    img = np.array(img)  # Ensure img is a numpy array
    img_clone = img.copy()
    noise = np.random.normal(0, sigma, img.shape)
    max_img = np.max(img)  # Ensure this is a scalar
    img_clone = img_clone + max_img * noise
    return img_clone

def create_noisy_sinograms(images, angles_full, sigma):
    # 0.1: Make geometry:
    geo = ctgeo.Geometry.parallel_default_parameters(
        image_shape=images.shape, number_of_angles=angles_full
    )  # parallel beam standard CT
    # 0.2: create operator:
    op = ct.make_operator(geo)
    # 0.3: forward project:
    sino = op(torch.from_numpy(images))
    sinogram_full = torch.tensor(add_gaussian_noise(sino, sigma))
    sinogram_full = torch.moveaxis(sinogram_full, -1, -2)
    return np.asarray(sinogram_full)


def addGaussNoise(image, mean=0, var=.01):
    row, col, ch = image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy


# @title Define function to preprocess and save


