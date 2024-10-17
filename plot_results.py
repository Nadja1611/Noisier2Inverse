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
parser.add_argument("-dataset", "--dataset", type=str, help="which dataset should be used", default="Heart")
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
parser.add_argument("-lr","--learning_rate",type=float,help="which learning rate should be used", default=1e-5)

args = parser.parse_args()

# Create output directory if not exists
output_dir = os.path.join(args.outputdir, "Test_Results")




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



# Set up the path where the results are stored
if args.dataset == 'Heart':
    path = '/home/nadja/tomo_project/Results_Noisier2Inverse_Heart/Test_Results'
else:
     path = '/home/nadja/tomo_project/Results_Noisier2Inverse/Test_Results'
files = os.listdir(path)

# Initialize lists to store data for plotting
noisy_list, clean_list = [], []
result_sob_y_list, result_sob_z_list = [], []
result_inf_y_list, result_inf_z_list = [], []
result_y_list, result_z_list = [], []
print(files, flush = True)
# Loop through files to load the appropriate data
for f in files:
    if 'Sob' in f and str(args.noise_sigma) in f:
        data = np.load(os.path.join(path, f, 'output_reco_results_z.npz'))  
        result_sob_z = data['output_reco_array']
        clean = data['clean_test']
        noisier = data['recos_test_z']
        noisy = data['recos_test_y']
        data = np.load(os.path.join(path, f, 'output_reco_results_y.npz'))
        result_sob_y = data['output_reco_array']

        ssim_values_sob_z = np.load(os.path.join(path, f,  'ssim_z.npy'))  # Changed ssim to ssim_values
        psnr_values_sob_z = np.load(os.path.join(path, f,  'psnr_z.npy'))
        emd_values_sob_z = np.load(os.path.join(path, f, 'emd_z.npy'))

        ssim_values_sob_y = np.load(os.path.join(path, f,  'ssim_y.npy'))  # Changed ssim to ssim_values
        psnr_values_sob_y = np.load(os.path.join(path, f,  'psnr_y.npy'))
        emd_values_sob_y = np.load(os.path.join(path, f, 'emd_y.npy'))


    elif 'Inference' in f and str(args.noise_sigma) in f and 'Sob' not in f:  

        data = np.load(os.path.join(path, f, 'output_reco_results_z.npz'))
        result_inf_z = data['output_reco_array']
        clean = data['clean_test']
        noisier = data['recos_test_z']
        noisy = data['recos_test_y']
        data = np.load(os.path.join(path, f, 'output_reco_results_y.npz'))
        result_inf_y = data['output_reco_array']

        ssim_values_inf_z = np.load(os.path.join(path, f,  'ssim_z.npy'))  # Changed ssim to ssim_values
        psnr_values_inf_z = np.load(os.path.join(path, f,  'psnr_z.npy'))
        emd_values_inf_z = np.load(os.path.join(path, f, 'emd_z.npy'))

        ssim_values_inf_y = np.load(os.path.join(path, f,  'ssim_y.npy'))  # Changed ssim to ssim_values
        psnr_values_inf_y = np.load(os.path.join(path, f,  'psnr_y.npy'))
        emd_values_inf_y = np.load(os.path.join(path, f, 'emd_y.npy'))

    elif str(args.noise_sigma) in f and 'Inf' not in f and 'Sob' not in f:
        print(f + ' we load inference weights for ' + str(args.noise_sigma))
        data = np.load(os.path.join(path, f, 'output_reco_results_z.npz'))
        result_z = data['output_reco_array']
        clean = data['clean_test']
        noisier = data['recos_test_z']
        noisy = data['recos_test_y']
        data = np.load(os.path.join(path, f, 'output_reco_results_y.npz'))
        result_y = data['output_reco_array']

        ssim_values_z = np.load(os.path.join(path, f,  'ssim_z.npy'))  # Changed ssim to ssim_values
        psnr_values_z = np.load(os.path.join(path, f,  'psnr_z.npy'))
        emd_values_z = np.load(os.path.join(path, f, 'emd_z.npy'))

        ssim_values_y = np.load(os.path.join(path, f,  'ssim_y.npy'))  # Changed ssim to ssim_values
        psnr_values_y = np.load(os.path.join(path, f,  'psnr_y.npy'))
        emd_values_y = np.load(os.path.join(path, f, 'emd_y.npy'))        

# Now that we have all the data loaded, let's create the plot
fig, axes = plt.subplots(5, 8, figsize=(20, 10))  # 5 rows, 8 columns

# Plot the data in the specified order for each row
for i in range(5):  # Assuming 5 rows
    # Column 1: Noisy data
    axes[i, 0].imshow(noisy[i], cmap='gray')
    axes[i, 0].set_title('Noisy')
    axes[i, 0].axis('off')
    
    # Column 2: Clean data
    axes[i, 1].imshow(clean[i], cmap='gray')
    axes[i, 1].set_title('Clean')
    axes[i, 1].axis('off')
    
    # Column 3: Result Sob Z
    axes[i, 2].imshow(result_sob_z[i], cmap='gray')
    axes[i, 2].set_title('Result Sob Z')
    axes[i, 2].axis('off')
    
    # Column 4: Result Sob Y
    axes[i, 3].imshow(result_sob_y[i], cmap='gray')
    axes[i, 3].set_title('Result Sob Y')
    axes[i, 3].axis('off')
    
    # Column 5: Result Inf Z
    axes[i, 4].imshow(result_inf_z[i], cmap='gray')
    axes[i, 4].set_title('Result Inf Z')
    axes[i, 4].axis('off')
    
    # Column 6: Result Inf Y
    axes[i, 5].imshow(result_inf_y[i], cmap='gray')
    axes[i, 5].set_title('Result Inf Y')
    axes[i, 5].axis('off')
    
    # Column 7: Result Y
    axes[i, 6].imshow(result_y[i], cmap='gray')
    axes[i, 6].set_title('Result Y')
    axes[i, 6].axis('off')
    
    # Column 8: Result Z
    axes[i, 7].imshow(result_z[i], cmap='gray')
    axes[i, 7].set_title('Result Z')
    axes[i, 7].axis('off')

# Adjust layout for better spacing
plt.tight_layout()
if args.dataset == 'Heart':
    plt.savefig(os.path.join('/home/nadja/tomo_project/Results_Noisier2Inverse_Heart/Plots_Paper', 'Plot' + str(args.noise_sigma) +'.png'))
else:
    plt.savefig(os.path.join('/home/nadja/tomo_project/Results_Noisier2Inverse/Plots_Paper', 'Plot' + str(args.noise_sigma) +'.png'))




plt.figure(figsize=(12, 6))

# Plot PSNR values
plt.subplot(1, 2, 1)
plt.plot(psnr_values_sob_z, label='PSNR Sob Z')
plt.plot(psnr_values_sob_y, label='PSNR Sob Y')
plt.plot(psnr_values_inf_z, label='PSNR Inf Z')
plt.plot(psnr_values_inf_y, label='PSNR Inf Y')
plt.plot(psnr_values_z, label='PSNR Z')
plt.plot(psnr_values_y, label='PSNR Y')
plt.title('PSNR Values sigma ' + str(args.noise_sigma))
plt.xlabel('Index')
plt.ylabel('PSNR (dB)')
plt.legend()

# Plot SSIM values
plt.subplot(1, 2, 2)
plt.plot(ssim_values_sob_z, label='SSIM Sob Z')
plt.plot(ssim_values_sob_y, label='SSIM Sob Y')
plt.plot(ssim_values_inf_z, label='SSIM Inf Z')
plt.plot(ssim_values_inf_y, label='SSIM Inf Y')
plt.plot(ssim_values_z, label='SSIM Z')
plt.plot(ssim_values_y, label='SSIM Y')
plt.title('SSIM Values sigma ' + str(args.noise_sigma))
plt.xlabel('Index')
plt.ylabel('SSIM')
plt.legend()
if args.dataset == 'Heart':
    plt.savefig(os.path.join('/home/nadja/tomo_project/Results_Noisier2Inverse_Heart/Plots_Paper/', 'Plot_ssim_psnr_' + str(args.noise_sigma) +'.png'))
else:
    plt.savefig(os.path.join('/home/nadja/tomo_project/Results_Noisier2Inverse/Plots_Paper/', 'Plot_ssim_psnr_' + str(args.noise_sigma) +'.png'))
