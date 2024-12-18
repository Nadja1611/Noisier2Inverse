import os
import numpy as np
import cv2 as cv
from skimage.transform import rescale, resize
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import gc
import matplotlib.pyplot as plt
import LION.CTtools.ct_geometry as ctgeo

from utils import *
from torch.utils.data import Dataset
import torchvision.transforms as transforms






#%% function for reading in our walnut data
def get_images(path, amount_of_images='all', scale_number=1):
    all_images = []
    all_image_names = os.listdir(path)
    if amount_of_images == 'all':
        for name in all_image_names:
            temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            image = temp_image[90:410, 90:410]
            image = image[0:320:scale_number, 0:320:scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)
    else:
        temp_indexing = np.random.permutation(len(all_image_names))[:amount_of_images]
        
        images_to_take = [all_image_names[i] for i in temp_indexing]
        for name in images_to_take:
            temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            image = temp_image[90:410, 90:410]
            image = image[0:320:scale_number, 0:320:scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)
    
    return all_images


class Walnut(Dataset):
    def __init__(self, data_dir="/home/nadja/tomo_project/Data/"
, noise_type='salt_and_pepper', noise_intensity=0.05, angles = 512, train=True, transform=None):
        super(Walnut, self).__init__()

        self.noise_intensity = noise_intensity
        self.noise_type = noise_type
        self.angles = angles
 
        if train:
            self.clean_dir = os.path.join(data_dir, 'train')
        else:
            self.clean_dir = os.path.join(data_dir, 'test')


        self.clean_paths = sorted(make_dataset(self.clean_dir))

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
                    #### add salt and peppern noise to given images 
    def add_salt_and_pepper_noise(self, image, salt_ratio, pepper_ratio): 
        """
        Adds salt and pepper noise to an image.

        Args:
            image (numpy.ndarray): Input image.
            salt_ratio (float): Ratio of salt noise (default: 0.05).
            pepper_ratio (float): Ratio of pepper noise (default: 0.05).

        Returns:
            numpy.ndarray: Image with salt and pepper noise.
        """
        pepper_ratio=self.noise_intensity
        salt_ratio=self.noise_intensity
        row, col = image.shape
        salt = np.random.rand(row, col) < salt_ratio
        pepper = np.random.rand(row, col) < pepper_ratio
        noisy_image = np.copy(image)
        noisy_image[salt] = 1
        noisy_image[pepper] = 0
        return noisy_image        
    
    def normalize(self, image):
        for i in range(len(image)):
            image[i] = image[i] - np.min(image[i])
            image[i] = image[i]/((np.max(image[i])+1e-5))
        return image   

    def __getitem__(self, index):
        clean_path = self.clean_paths[index]
        clean = np.array(cv.imread(clean_path, cv.IMREAD_GRAYSCALE))
        
        
        if self.noise_type == 'gauss':
            """ In that case, we add the gaussian noise in the sinogram, then noise is uncorrelated in sino domain """
            clean = np.array([img.squeeze() for img in clean], dtype='float16')
            clean = clean/np.max(clean)
            print(clean.shape)
            clean_sino = np.array((create_noisy_sinograms(np.expand_dims(clean,0), self.angles, 0)).squeeze(0))
            print(clean_sino.shape)
            noisy = np.asarray(add_gaussian_noise(clean_sino, self.noise_intensity))


        if self.noise_type == 'gauss_image':
            """ In that case, we add the gaussian noise in the sinogram, then noise is uncorrelated in sino domain """
            clean = np.array([img.squeeze() for img in clean], dtype='float16')
            clean = clean/np.max(clean)
            print(clean.shape)
            clean = np.array([img.squeeze() for img in clean], dtype='float16')
            clean = clean/np.max(clean)
            clean_sino = np.array((create_noisy_sinograms(np.expand_dims(clean,0), self.angles, 0)).squeeze(0))
            print(clean_sino.shape)
            noisy = np.asarray(add_gaussian_noise(clean_sino, self.noise_intensity))


        else:
            raise NotImplementedError('wrong type of noise 2')

        if self.noise_type == 'gauss':
            clean, noisy = self.transform(clean),  self.transform(noisy)

        else:
            raise NotImplementedError('wrong type of noise')

        if self.noise_type == 'gauss':
            return {'clean': clean,  'noisy': noisy}

        else:
            raise NotImplementedError('wrong type of noise ')
    def __len__(self):
        return len(self.clean_paths)















