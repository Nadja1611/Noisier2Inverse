# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:51:34 2024

@author: nadja
"""

# %%
import torch
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import sys
import skimage.metrics as skm
from skimage.data import shepp_logan_phantom
import logging
import numpy as np
from tomosipo.torch_support import (
    to_autograd,
)
from itertools import combinations

from tqdm import tqdm
from model import *
from itertools import combinations
import LION.CTtools.ct_utils as ct
from ts_algorithms import fbp, tv_min2d
import skimage
import argparse
import gc
from scipy.ndimage import gaussian_filter
from dataset import *
from torch.utils.tensorboard import SummaryWriter

# %%
parser = argparse.ArgumentParser(
    description="Arguments for segmentation network.", add_help=False
)
parser.add_argument(
    "-l",
    "--loss_variant",
    type=str,
    help="which loss variant should be used",
    default="DataDomain_NW_Data_MSE",
)
parser.add_argument(
    "-angles",
    "--angles",
    type=int,
    help="number of prosqueuejection angles sinogram",
    default=512,
)
parser.add_argument(
    "-batch_size",
    "--batch_size",
    type=int,
    help="what batch size should be used",
    default=6,
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    help="which learning rate should be used",
    default=1e-4,
)
parser.add_argument(
    "-noise_type",
    "--noise_type",
    type=str,
    help="add correlated or uncorrelated noise",
    default="gauss",
)

parser.add_argument(
    "-o",
    "--logdir",
    type=str,
    help="directory for log files",
    default="/home/nadja/tomo_project/LION/logs",
)


parser.add_argument(
    "-datadir",
    "--datadir",
    type=str,
    help="from where should the data be loaded",
    default="/home/nadja/tomo_project/Data_CT",
)

parser.add_argument(
    "-noise_intensity",
    "--noise_intensity",
    type=float,
    help="how intense is the gaussian/poisson noise",
    default=1.0,
)
parser.add_argument(
    "-noise_sigma",
    "--noise_sigma",
    type=float,
    help="how big is the kernel size of convolution",
    default=3.0,
)


parser.add_argument(
    "-out",
    "--outputdir",
    type=str,
    help="directory where results are saved",
    default="/home/nadja/tomo_project/Results_Noise2Inverse_Heart_vsc/",
)

args = parser.parse_args()

# Create output directory if not exists
output_dir = os.path.join(args.outputdir, "Test_Results")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


output_dir = (
    output_dir
    + "/N2I_"
    + args.noise_type
    + "noise_sigma_"
    + str(args.noise_sigma)
    + "_"
    + str(args.noise_intensity)
    + "_"
    + "batch_size_"
    + str(args.batch_size)
    + "_"
    + "Gaussian_Method_"
    + args.loss_variant
    + "_learning_rate_"
    + str(args.learning_rate)
    + "_angles_"
    + str(args.angles)
)


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

""" explanation loss_variant"""
## DataDomain_MSE         - loss is computed in data domain, NW operates in recon domain (sparse2inverse)
## ReconDomain_MSE        - loss is computed in recon domain, NW operates in recon domain
## DataDomain_Sobolev     - Sobolev loss is computed in data domain, NW operates in recon domain

weights_dir = (
    args.outputdir
    + "/Model_Weights/N2I_"
    + args.noise_type
    + "noise_sigma_"
    + str(args.noise_sigma)
    + "_"
    + str(args.noise_intensity)
    + "_"
    + "batch_size_"
    + str(args.batch_size)
    + "_"
    + "Gaussian_Method_"
    + args.loss_variant
    + "_learning_rate_"
    + str(args.learning_rate)
    + "_angles_"
    + str(args.angles)
)

if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)


device = "cpu"


# %%

photon_count = 100  #
attenuation_factor = 2.76  # corresponds to absorption of 50%


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


def create_noisy_sinograms(images, angles_full, sigma):
    # 0.1: Make geometry:
    geo = ctgeo.Geometry.parallel_default_parameters(
        image_shape=images.shape, number_of_angles=angles_full
    )  # parallel beam standard CT
    # 0.2: create operator:
    op = ct.make_operator(geo)
    # 0.3: forward project:
    sino = op(torch.from_numpy(images))
    # sinogram_full = add_gaussian_noise(sino, sigma)
    sinogram_full = torch.moveaxis(sino, -1, -2)
    return np.asarray(sinogram_full.unsqueeze(1))


class Noise2Inverse:
    def __init__(
        self,
        device: str = "cpu",
        folds: int = 4,
    ):
        self.net_denoising = UNet(in_channels=1, out_channels=1).to(device)
        self.folds = folds
        self.device = device
        self.angles = args.angles
        self.batch_size = args.batch_size

        # speicift noise type and intensity
        self.noise = args.noise_type
        self.noise_intensity = args.noise_intensity

        self.test_dataset = Walnut_test(
            noise_type=self.noise,
            noise_intensity=self.noise_intensity,
            noise_sigma=args.noise_sigma,
            train=False,
            data_dir=args.datadir,
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def forward(self, reconstruction, nr_angles):
        input_x = reconstruction
        output_denoising = self.net_denoising(input_x.float().to(self.device))
        # with torch.no_grad():
        output_denoising_sino = self.projection_tomosipo(
            output_denoising, sino=nr_angles
        ).to(self.device)
        return output_denoising, output_denoising_sino

    def prepare_batch(self, sinograms):
        sinograms = sinograms.squeeze()
        Reconstructions = torch.zeros(
            (sinograms.shape[0], self.folds, sinograms.shape[-2], sinograms.shape[-2])
        ).to(sinograms.device)
        number_of_angles = sinograms.shape[-1]
        projection_indices = np.array([i for i in range(0, number_of_angles)])
        indices_four_folds = [
            projection_indices[i :: self.folds] for i in range(self.folds)
        ]
        sinograms_four_folds = sinograms[:, :, indices_four_folds]
        sinograms_four_folds = torch.movedim(sinograms_four_folds, -2, 1)

        for i in range(sinograms.shape[0]):
            for j in range(self.folds):
                Reconstructions[i, j] = self.fbp_tomosipo(
                    torch.tensor(sinograms_four_folds[i][j].unsqueeze(0).unsqueeze(0)),
                    translate="N2I",
                    angle_vector=indices_four_folds[j],
                    folds=self.folds,
                )
        return (Reconstructions), indices_four_folds

    def projection_tomosipo(self, img, sino, translate=False):
        if sino.dtype == int:
            angles = sino
        else:
            angles = sino.shape[-1]
        # 0.1: Make geometry:
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(sino.shape[0], 336, 336),
            number_of_angles=angles,
            translate=translate,
        )
        # 0.2: create operator:
        op = to_autograd(ct.make_operator(geo))
        sino = op((img[:, 0]).to(self.device))
        sino = sino.unsqueeze(1)
        sino = torch.moveaxis(sino, -1, -2)
        return sino

    def fbp_tomosipo(self, sino, angle_vector=None, translate=False, folds=None):
        angles = sino.shape[-1]
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(sino.shape[0], 336, 336),
            number_of_angles=angles,
            translate=translate,
            angle_vector=angle_vector,
        )
        op = ct.make_operator(geo)
        sino = torch.moveaxis(sino, -1, -2)
        result = fbp(op, sino[:, 0])
        result = result.unsqueeze(1)
        return result


#### generate a new output path where the results are stored!
newpath = (
    args.outputdir
    + "/N2I_"
    + args.noise_type
    + "noise_sigma_"
    + str(args.noise_sigma)
    + "_"
    + str(args.noise_intensity)
    + "_"
    + "batch_size_"
    + str(args.batch_size)
    + "_"
    + "Gaussian_Method_"
    + args.loss_variant
    + "_learning_rate_"
    + str(args.learning_rate)
    + "_angles_"
    + str(args.angles)
)
if not os.path.exists(newpath):
    os.makedirs(newpath)


###### specifiy training parameters
learning_rate = args.learning_rate

output_reco_list = []
clean_list = []
noisy_list = []
###### Choose from 'MSE_image', 'MSE_data', 'Sobolev_data'
N2I = Noise2Inverse()
# Initialize testing class

weights_path = os.path.join(weights_dir, "psnr_model_weights_.pth")

if not os.path.exists(weights_path):
    print("File not found, using default weights.")
    weights_path = os.path.join(weights_dir, "psnr_model_weights.pth")


print(weights_path, flush=True)
N2I.net_denoising.load_state_dict(torch.load(weights_path))
N2I.net_denoising.eval()


########################### Now training starts ##############
l2_list = []
all_MSEs = []
# Initialize empty tensors for accumulating mean values
all_ssim = torch.tensor([])
all_psnr = torch.tensor([])

# Initialize old ssim and psnr
old_ssim = 0.1
old_psnr = 0.1


MSEs = []
ssim_y = []
psnr_y = []

with torch.no_grad():
    N2I.net_denoising.eval()

    for batch, data in enumerate(N2I.test_dataloader):
        #### apply gassian noise a second time to make the sinogram even noisier
        clean_test, noisy_test = (
            data["clean"].squeeze(),
            data["noisy"].squeeze(),
        )
        y_test = noisy_test.unsqueeze(1).to(N2I.device)
        # y = torch.moveaxis(y, -1,-2)
        y_recos_test, indices_test = N2I.prepare_batch(y_test)

        subsets = combinations(range(4), 3)
        final_recos = torch.zeros(
            (
                y_recos_test.shape[0],
                1,
                y_recos_test.shape[2],
                y_recos_test.shape[3],
            )
        ).to(N2I.device)

        for rand_ints in subsets:
            input_x_den = torch.mean(y_recos_test[:, rand_ints], axis=1)
            input_x_den = input_x_den.unsqueeze(1)
            # target is \tilde{x}_J, as |J| = 1 , no mean required
            # divide by 5 for faster NW convergence
            input_x_den = input_x_den / 5
            output_reco, output_sino = N2I.forward(input_x_den.to(N2I.device), y_test)
            final_recos += output_reco * 5 / 4

        for i in range(len(clean_test)):
            # Ensure the tensors are on CPU and converted to numpy arrays
            ims_test_np = clean_test[i].detach().cpu().numpy()
            output_reco_y_np = final_recos[i][0].detach().cpu().numpy()

            # Calculate the data range for SSIM and PSNR
            data_range = ims_test_np.max() - ims_test_np.min()

            # Compute SSIM for y and z
            ssim_y_value = torch.tensor(
                skimage.metrics.structural_similarity(
                    output_reco_y_np, ims_test_np, data_range=data_range
                )
            ).to(device)

            # Compute PSNR for y and z
            psnr_y_value = torch.tensor(
                skimage.metrics.peak_signal_noise_ratio(
                    output_reco_y_np, ims_test_np, data_range=data_range
                )
            ).to(device)

            # Append the computed values to the respective lists
            ssim_y.append(ssim_y_value)
            psnr_y.append(psnr_y_value)
            output_reco_list.append(output_reco_y_np)
            clean_list.append(ims_test_np)
            noisy_list.append(output_reco_y_np)


output_reco_array = np.array(output_reco_list)
clean_array = np.array(clean_list)
recos_test_y_array = np.array(noisy_list)


ssim_array = np.array(ssim_y)
psnr_array = np.array(psnr_y)
print(os.path.join(output_dir, "output_reco_results.npz"), flush=True)
np.savez_compressed(
    os.path.join(output_dir, "output_reco_results.npz"),
    output_reco_array=output_reco_array,
    clean_test=clean_array,
    recos_test_y=recos_test_y_array,
)
np.save(os.path.join(output_dir, "ssim_z.npy"), ssim_array)
np.save(os.path.join(output_dir, "psnr_z.npy"), psnr_array)
