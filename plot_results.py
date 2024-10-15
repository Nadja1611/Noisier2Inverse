# -*- coding: utf-8 -*-
"""
Script for plotting reconstructions and metrics (SSIM, PSNR, EMD)
"""

# %%
import torch
import os
import matplotlib.pyplot as plt
import skimage.metrics as skm
from skimage.data import shepp_logan_phantom
import logging
import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from model import *
from dataset_EMD import *
from utils_inverse import create_noisy_sinograms
import argparse
from torch.utils.data import DataLoader
import psutil
from ts_algorithms import fbp, tv_min2d


# Parsing arguments for testing
parser = argparse.ArgumentParser(description="Arguments for testing denoising network.", add_help=False)
parser.add_argument("-l", "--loss_variant", type=str, help="which loss variant should be used", default="DataDomain_NW_Data_MSE")
parser.add_argument("-l2", "--loss_variant2", type=str, help="which loss variant should be used for comparison", default="DataDomain_NW_Data_MSE")

parser.add_argument("-alpha", "--alpha", type=float, help="how much noisier should z be than y", default=1)
parser.add_argument("-angles", "--angles", type=int, help="number of projection angles sinogram", default=512)
parser.add_argument("-batch_size", "--batch_size", type=int, help="batch size for testing", default=6)
parser.add_argument("-datadir", "--datadir", type=str, help="directory for data loading", default='/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data/')
parser.add_argument("-noise_type", "--noise_type", type=str, help="type of noise", default="uncorrelated")
parser.add_argument("-noise_intensity", "--noise_intensity", type=float, help="intensity of salt and pepper noise", default=0.05)
parser.add_argument("-out", "--outputdir", type=str, help="directory for saving results", default="/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Test/")
parser.add_argument("-w","--weights_dir",
    type=str,
    help="directory to save model weights",
    default="/home/nadja/tomo_project/Results_Noisier2Inverse_Heart/Model_Weights",
)
parser.add_argument(
    "-noise_sigma",
    "--noise_sigma",
    type=float,
    help="how big is the kernel size of convolution",
    default=3.0,
)
parser.add_argument("-y_z", "--dat", type=str, help="predict on y or on z?", default="z")
parser.add_argument("-lr","--learning_rate",type=float,help="which learning rate should be used", default=1e-5)
parser.add_argument("-lr2","--learning_rate2",type=float,help="which learning rate should be used in second method", default=1e-5)

args = parser.parse_args()

# Create output directory if not exists
output_dir = os.path.join(args.outputdir, "Test_Results")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


output_dir =(
    output_dir
   + "/Noise_"
    + args.noise_type
    + "_"
    + str(args.noise_intensity)
        + "_sigma_"
    + str(args.noise_sigma)
    + "_batchsize_"
    + str(args.batch_size)
    + "_"
    + "Gaussian_Method_"
    + args.loss_variant
    + "_alpha_"
    + str(args.alpha)
    + "_learning_rate_"
    + str(args.learning_rate)
    + "_angles_"
    + str(args.angles)
)


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

"""specify weight directory"""
weights_dir = (
    args.weights_dir
    + "/Noise_"
    + args.noise_type
    + "_"
    + str(args.noise_intensity)
        + "_sigma_"
    + str(args.noise_sigma)
    + "_batchsize_"
    + str(args.batch_size)
    + "_"
    + "Gaussian_Method_"
    + args.loss_variant
    + "_alpha_"
    + str(args.alpha)
    + "_learning_rate_"
    + str(args.learning_rate)
    + "_angles_"
    + str(args.angles)
)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


weights_path = os.path.join(weights_dir, "emd_model_weights_"+ args.dat + ".pth")  # Assuming the best weights were saved


# Save the arrays as .npy files
if "emd_model_weights_y.pth" in weights_path:
    data = np.load(
    os.path.join(output_dir, 'output_reco_results_y.npz')
    )


# Save the arrays as .npy files
if "emd_model_weights_z.pth" in weights_path:
    data = np.load(
    os.path.join(output_dir, 'output_reco_results_z.npz')
    )   


# Access each array by its name
reconstructed = data['output_reco_array']
clean = data['clean_test']
noisier = data['recos_test_z']
noisy = data['recos_test_y']
print(clean.shape, flush = True)
print(noisy.shape, flush = True)
print(reconstructed.shape, flush = True)
# %%
# Function to plot reconstructions (e.g., clean, noisy, and reconstructed images)
def plot_reconstructions(clean, noisy, reconstructed, index, save_path):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(clean, cmap='gray')
    plt.title('Clean Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed, cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')

    plt.suptitle(f'Reconstruction Comparison (Index: {index})')
    
    plt.savefig(os.path.join(save_path, f'reconstruction_{index}.png'))
    plt.show()

# %%


# Load SSIM, PSNR, and EMD arrays
if "emd_model_weights_z.pth" in weights_path:
    ssim_values = np.load(os.path.join(output_dir, 'ssim_z.npy'))  # Changed ssim to ssim_values
    psnr_values = np.load(os.path.join(output_dir, 'psnr_z.npy'))
    emd_values = np.load(os.path.join(output_dir, 'emd_z.npy'))

# Load SSIM, PSNR, and EMD arrays
if "emd_model_weights_y.pth" in weights_path:
    ssim_values = np.load(os.path.join(output_dir, 'ssim_y.npy'))  # Changed ssim to ssim_values
    psnr_values = np.load(os.path.join(output_dir, 'psnr_y.npy'))
    emd_values = np.load(os.path.join(output_dir, 'emd_y.npy'))

# Function to plot the SSIM, PSNR, and EMD metrics as curves
def plot_metrics(ssim_values, psnr_values, save_path):
    iterations = np.arange(len(ssim_values))
    mean_ssim = np.mean(ssim_values)
    mean_psnr = np.mean(psnr_values)
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(3, 1, 1)
    plt.plot(iterations, ssim_values, label='SSIM', color='blue', marker='o')
    plt.title('SSIM Over Test Images, Mean: '+ str(mean_ssim))
    plt.xlabel('Test Image Index')
    plt.ylabel('SSIM')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(iterations, psnr_values, label='PSNR', color='green', marker='x')
    plt.title('PSNR Over Test Images, Mean: '+ str(mean_psnr))
    plt.xlabel('Test Image Index')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(iterations, emd_values, label='EMD', color='red', marker='*')
    plt.title('EMD Over Test Images')
    plt.xlabel('Test Image Index')
    plt.ylabel('EMD')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics_curves.png'))
    plt.show()

# %%
# Path to save plots
output_plot_dir = os.path.join(output_dir, 'Plots')
if not os.path.exists(output_plot_dir):
    os.makedirs(output_plot_dir)

def plot_reconstructions_grid(clean, noisy, reconstructed, save_path, num_images=5):
    rows, cols = num_images, 3  # Each row will contain clean, noisy, and reconstructed images for the same index
    fig, axs = plt.subplots(rows, cols, figsize=(5, 3 * num_images))  # Adjust the figure size accordingly
    
    for i in range(num_images):
        # Plot Clean Image in column 1
        axs[i, 0].imshow(clean[i], cmap='gray')
        axs[i, 0].set_title('Clean Image')
        axs[i, 0].axis('off')

        # Plot Noisy Image in column 2
        if len(noisy[i].shape) == 2:  # If noisy image has only 2 dimensions (height, width)
            axs[i, 1].imshow(noisy[i], cmap='gray')
        elif len(noisy[i].shape) == 3:  # If noisy image has 3 dimensions (channels, height, width)
            axs[i, 1].imshow(noisy[i][0], cmap='gray')  # Adjust based on actual structure
        axs[i, 1].set_title('Noisy Image')
        axs[i, 1].axis('off')

        # Plot Reconstructed Image in column 3
        if len(reconstructed[i].shape) == 2:
            axs[i, 2].imshow(reconstructed[i], cmap='gray')
        elif len(reconstructed[i].shape) == 3:
            axs[i, 2].imshow(reconstructed[i][0], cmap='gray')  # Adjust if needed
        axs[i, 2].set_title('Reconstructed Image')
        axs[i, 2].axis('off')

    plt.suptitle(f'Reconstruction Comparison (First {num_images} Images)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    plt.savefig(os.path.join(save_path, f'reconstructions_grid.png'))
    plt.show()



plot_reconstructions_grid(clean, noisy, reconstructed, output_plot_dir, num_images=5)
plot_metrics(ssim_values, psnr_values, output_plot_dir)
