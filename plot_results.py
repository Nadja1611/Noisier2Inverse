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
parser = argparse.ArgumentParser(
    description="Arguments for testing denoising network.", add_help=False
)
parser.add_argument(
    "-l",
    "--loss_variant",
    type=str,
    help="which loss variant should be used",
    default="DataDomain_NW_Data_MSE",
)
parser.add_argument(
    "-dataset",
    "--dataset",
    type=str,
    help="which dataset should be used",
    default="Heart",
)
parser.add_argument(
    "-alpha",
    "--alpha",
    type=float,
    help="how much noisier should z be than y",
    default=1,
)
parser.add_argument(
    "-angles",
    "--angles",
    type=int,
    help="number of projection angles sinogram",
    default=512,
)
parser.add_argument(
    "-batch_size", "--batch_size", type=int, help="batch size for testing", default=6
)
parser.add_argument(
    "-datadir",
    "--datadir",
    type=str,
    help="directory for data loading",
    default="/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data/",
)
parser.add_argument(
    "-noise_type",
    "--noise_type",
    type=str,
    help="type of noise",
    default="uncorrelated",
)
parser.add_argument(
    "-noise_intensity",
    "--noise_intensity",
    type=float,
    help="intensity of salt and pepper noise",
    default=0.05,
)
parser.add_argument(
    "-out",
    "--outputdir",
    type=str,
    help="directory for saving results",
    default="/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Test/",
)
parser.add_argument(
    "-w",
    "--weights_dir",
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
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    help="which learning rate should be used",
    default=1e-5,
)

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
if args.dataset == "Heart":
    path = "/home/nadja/tomo_project/Results_Noisier2Inverse_Heart/Test_Results"
    path2 = "/home/nadja/tomo_project/Results_Noise2Inverse_Heart_vsc/Test_Results/"
else:
    path = "/home/nadja/tomo_project/Results_Noisier2Inverse/Test_Results"
    path2 = "/home/nadja/tomo_project/Results_Noise2Inverse_vsc/Test_Results/"

files = os.listdir(path)
files_n2i = os.listdir(path2)

# Initialize lists to store data for plotting
noisy_list, clean_list = [], []
result_sob_y_list, result_sob_z_list = [], []
result_inf_y_list, result_inf_z_list = [], []
result_y_list, result_z_list = [], []
print(files, flush=True)
# Loop through files to load the appropriate data
for f in files:
    print(f, flush=True)
    if "Sob" in f and "sigma_" + str(args.noise_sigma) in f:
        data = np.load(os.path.join(path, f, "output_reco_results_z.npz"))
        result_sob_z = data["output_reco_array"]
        clean = data["clean_test"]
        noisier = data["recos_test_z"]
        noisy = data["recos_test_y"]
        data = np.load(os.path.join(path, f, "output_reco_results_y.npz"))
        result_sob_y = data["output_reco_array"]

        ssim_values_sob_z = np.load(
            os.path.join(path, f, "ssim_z.npy")
        )  # Changed ssim to ssim_values
        psnr_values_sob_z = np.load(os.path.join(path, f, "psnr_z.npy"))
        emd_values_sob_z = np.load(os.path.join(path, f, "emd_z.npy"))

        ssim_values_sob_y = np.load(
            os.path.join(path, f, "ssim_y.npy")
        )  # Changed ssim to ssim_values
        psnr_values_sob_y = np.load(os.path.join(path, f, "psnr_y.npy"))
        emd_values_sob_y = np.load(os.path.join(path, f, "emd_y.npy"))

    elif "Inference" in f and "sigma_" + str(args.noise_sigma) in f and "Sob" not in f:
        data = np.load(os.path.join(path, f, "output_reco_results_z.npz"))
        result_inf_z = data["output_reco_array"]
        clean = data["clean_test"]
        noisier = data["recos_test_z"]
        noisy = data["recos_test_y"]
        data = np.load(os.path.join(path, f, "output_reco_results_y.npz"))
        result_inf_y = data["output_reco_array"]

        ssim_values_inf_z = np.load(
            os.path.join(path, f, "ssim_z.npy")
        )  # Changed ssim to ssim_values
        psnr_values_inf_z = np.load(os.path.join(path, f, "psnr_z.npy"))
        emd_values_inf_z = np.load(os.path.join(path, f, "emd_z.npy"))

        ssim_values_inf_y = np.load(
            os.path.join(path, f, "ssim_y.npy")
        )  # Changed ssim to ssim_values
        psnr_values_inf_y = np.load(os.path.join(path, f, "psnr_y.npy"))
        emd_values_inf_y = np.load(os.path.join(path, f, "emd_y.npy"))

    elif "sigma_" + str(args.noise_sigma) in f and "Inf" not in f and "Sob" not in f:
        print(f + " we load inference weights for " + str(args.noise_sigma))
        data = np.load(os.path.join(path, f, "output_reco_results_z.npz"))
        result_z = data["output_reco_array"]
        clean = data["clean_test"]
        noisier = data["recos_test_z"]
        noisy = data["recos_test_y"]
        data = np.load(os.path.join(path, f, "output_reco_results_y.npz"))
        result_y = data["output_reco_array"]

        ssim_values_z = np.load(
            os.path.join(path, f, "ssim_z.npy")
        )  # Changed ssim to ssim_values
        psnr_values_z = np.load(os.path.join(path, f, "psnr_z.npy"))
        emd_values_z = np.load(os.path.join(path, f, "emd_z.npy"))

        ssim_values_y = np.load(
            os.path.join(path, f, "ssim_y.npy")
        )  # Changed ssim to ssim_values
        psnr_values_y = np.load(os.path.join(path, f, "psnr_y.npy"))
        emd_values_y = np.load(os.path.join(path, f, "emd_y.npy"))


##### load results of noise2inverse
for f in files_n2i:
    if "sigma_" + str(args.noise_sigma) in f:
        print("we load n2i", flush=True)
        data = np.load(os.path.join(path2, f, "output_reco_results.npz"))
        result_n2i = data["output_reco_array"]
        ssim_values_n2i = np.load(
            os.path.join(path2, f, "ssim_z.npy")
        )  # Changed ssim to ssim_values
        psnr_values_n2i = np.load(os.path.join(path2, f, "psnr_z.npy"))


fig, axes = plt.subplots(
    5, 9, figsize=(25, 12), gridspec_kw={"wspace": 0.001, "hspace": 0.01}
)  # Further reduced spacing between images

"""------------make z plots--------------"""

# Define the titles for each column
titles = [
    "noisy ($\\sigma$ = " + str(args.noise_sigma) + ")",
    "clean",
    "ours sobo z",
    "ours z",
    "z",
    "N2I",
]

# Create a figure and axes with specified size and no spacing
fig, axes = plt.subplots(2, 6, figsize=(10, 3.5), constrained_layout=False)

# Adjust spacing to remove gaps between subplots
plt.subplots_adjust(wspace=0, hspace=0)  # Remove width and height space

# Plot the data in the specified order for each row
for i in range(2):  # Now only 3 rows
    for j in range(6):  # 9 columns
        # Column content based on j index, replacing i with i + 5
        if j == 0:
            axes[i, j].imshow(
                noisy[i + 5][40:-40, 40:-40], cmap="gray"
            )  # Adjusted index for noisy
        elif j == 1:
            axes[i, j].imshow(
                clean[i + 5][40:-40, 40:-40], cmap="gray", vmin=0, vmax=1
            )  # Adjusted index for clean
        elif j == 2:
            axes[i, j].imshow(
                result_sob_z[i + 5][40:-40, 40:-40],
                cmap="gray",
                vmin=0,
                vmax=1,  # Adjusted index for sobel z
            )
        elif j == 3:
            axes[i, j].imshow(
                result_inf_z[i + 5][40:-40, 40:-40],
                cmap="gray",
                vmin=0,
                vmax=1,  # Adjusted index for sobel y
            )
        elif j == 4:
            axes[i, j].imshow(
                result_z[i + 5][40:-40, 40:-40],
                cmap="gray",
                vmin=0,
                vmax=1,  # Adjusted index for inf z
            )
        elif j == 5:
            axes[i, j].imshow(
                result_n2i[i + 5][40:-40, 40:-40],
                cmap="gray",
                vmin=0,
                vmax=1,  # Adjusted index for inf z
            )

        # Remove axis for cleaner view
        axes[i, j].axis("off")

        # Set title only for the first row
        if i == 0:
            axes[i, j].set_title(titles[j], fontsize=12)

# Set the background color for better visual contrast
fig.patch.set_facecolor("white")
# Adjust layout for better spacing and save the figure
plt.tight_layout()

if args.dataset == "Heart":
    print("heart", flush=True)
    plt.savefig(
        os.path.join(
            "/home/nadja/tomo_project/Results_Noisier2Inverse_Heart/Plots_Paper",
            "Plot" + str(args.noise_sigma) + "_z.svg",
        )
    )
    plt.savefig(
        os.path.join(
            "/home/nadja/tomo_project/Results_Noisier2Inverse_Heart/Plots_Paper",
            "Plot" + str(args.noise_sigma) + "_z.png",
        )
    )
else:
    plt.savefig(
        os.path.join(
            "/home/nadja/tomo_project/Results_Noisier2Inverse/Plots_Paper",
            "Plot" + str(args.noise_sigma) + "_z.svg",
        )
    )
    plt.savefig(
        os.path.join(
            "/home/nadja/tomo_project/Results_Noisier2Inverse/Plots_Paper",
            "Plot" + str(args.noise_sigma) + "_z.png",
        )
    )


""" make y plots"""

# Define the titles for each column
titles = [
    "noisy ($\\sigma$ = " + str(args.noise_sigma) + ")",
    "clean",
    "ours sobo y",
    "ours y",
    "y",
    "N2I",
]

# Create a figure and axes with specified size and no spacing
fig, axes = plt.subplots(2, 6, figsize=(10, 3.5), constrained_layout=False)

# Adjust spacing to remove gaps between subplots
plt.subplots_adjust(wspace=0, hspace=0)  # Remove width and height space

# Plot the data in the specified order for each row
for i in range(2):  # Now only 3 rows
    for j in range(6):  # 9 columns
        # Column content based on j index, replacing i with i + 5
        if j == 0:
            axes[i, j].imshow(
                noisy[i + 5][40:-40, 40:-40], cmap="gray"
            )  # Adjusted index for noisy
        elif j == 1:
            axes[i, j].imshow(
                clean[i + 5][40:-40, 40:-40], cmap="gray", vmin=0, vmax=1
            )  # Adjusted index for clean
        elif j == 2:
            axes[i, j].imshow(
                result_sob_y[i + 5][40:-40, 40:-40],
                cmap="gray",
                vmin=0,
                vmax=1,  # Adjusted index for sobel z
            )
        elif j == 3:
            axes[i, j].imshow(
                result_inf_y[i + 5][40:-40, 40:-40],
                cmap="gray",
                vmin=0,
                vmax=1,  # Adjusted index for sobel y
            )
        elif j == 4:
            axes[i, j].imshow(
                result_y[i + 5][40:-40, 40:-40],
                cmap="gray",
                vmin=0,
                vmax=1,  # Adjusted index for inf z
            )
        elif j == 5:
            axes[i, j].imshow(
                result_n2i[i + 5][40:-40, 40:-40],
                cmap="gray",
                vmin=0,
                vmax=1,  # Adjusted index for inf z
            )

        # Remove axis for cleaner view
        axes[i, j].axis("off")

        # Set title only for the first row
        if i == 0:
            axes[i, j].set_title(titles[j], fontsize=12)

# Set the background color for better visual contrast
fig.patch.set_facecolor("white")
# Adjust layout for better spacing and save the figure
plt.tight_layout()

if args.dataset == "Heart":
    print("heart", flush=True)
    plt.savefig(
        os.path.join(
            "/home/nadja/tomo_project/Results_Noisier2Inverse_Heart/Plots_Paper",
            "Plot" + str(args.noise_sigma) + "_y.svg",
        )
    )
    plt.savefig(
        os.path.join(
            "/home/nadja/tomo_project/Results_Noisier2Inverse_Heart/Plots_Paper",
            "Plot" + str(args.noise_sigma) + "_y.png",
        )
    )
else:
    plt.savefig(
        os.path.join(
            "/home/nadja/tomo_project/Results_Noisier2Inverse/Plots_Paper",
            "Plot" + str(args.noise_sigma) + "_y.svg",
        )
    )
    plt.savefig(
        os.path.join(
            "/home/nadja/tomo_project/Results_Noisier2Inverse/Plots_Paper",
            "Plot" + str(args.noise_sigma) + "_y.png",
        )
    )


plt.figure(figsize=(12, 6))

# Plot PSNR values
plt.subplot(1, 2, 1)
plt.plot(psnr_values_sob_z, label="PSNR Sob z " + str(np.mean(psnr_values_sob_z)))
plt.plot(psnr_values_sob_y, label="PSNR Sob y" + str(np.mean(psnr_values_sob_y)))
plt.plot(psnr_values_inf_z, label="PSNR Inf z" + str(np.mean(psnr_values_inf_z)))
plt.plot(psnr_values_inf_y, label="PSNR Inf y" + str(np.mean(psnr_values_inf_y)))
plt.plot(psnr_values_z, label="PSNR z" + str(np.mean(psnr_values_inf_z)))
plt.plot(psnr_values_y, label="PSNR y" + str(np.mean(psnr_values_y)))
plt.plot(psnr_values_n2i, label="PSNR N2I" + str(np.mean(psnr_values_n2i)))
plt.title("PSNR Values sigma " + str(args.noise_sigma))
plt.xlabel("Index")
plt.ylabel("PSNR (dB)")
plt.legend()

# Plot SSIM values
plt.subplot(1, 2, 2)
plt.plot(ssim_values_sob_z, label="SSIM Sob z" + str(np.mean(ssim_values_sob_z)))
plt.plot(ssim_values_sob_y, label="SSIM Sob y" + str(np.mean(ssim_values_sob_y)))
plt.plot(ssim_values_inf_z, label="SSIM Inf z" + str(np.mean(ssim_values_inf_z)))
plt.plot(ssim_values_inf_y, label="SSIM Inf y" + str(np.mean(ssim_values_inf_y)))
plt.plot(ssim_values_z, label="SSIM z" + str(np.mean(ssim_values_z)))
plt.plot(ssim_values_y, label="SSIM y" + str(np.mean(ssim_values_y)))
plt.plot(ssim_values_n2i, label="SSIM N2I" + str(np.mean(ssim_values_n2i)))
plt.title("SSIM Values sigma " + str(args.noise_sigma))
plt.xlabel("Index")
plt.ylabel("SSIM")
plt.legend()


if args.dataset == "Heart":
    plt.savefig(
        os.path.join(
            "/home/nadja/tomo_project/Results_Noisier2Inverse_Heart/Plots_Paper/",
            "Plot_ssim_psnr_" + str(args.noise_sigma) + ".png",
        )
    )
else:
    plt.savefig(
        os.path.join(
            "/home/nadja/tomo_project/Results_Noisier2Inverse/Plots_Paper/",
            "Plot_ssim_psnr_" + str(args.noise_sigma) + ".png",
        )
    )


import matplotlib.pyplot as plt
import os

# Define vmin and vmax based on the dataset
if args.dataset == "Heart":
    vmin, vmax = -0.2, 0.2
else:
    vmin, vmax = -0.8, 0.8


# Function to create and save plots
def create_difference_plots(titles, result_list, noise_sigma, plot_type):
    # Create a larger figure and axes with specified size, adjusting for 4 columns
    figsize = (
        (18, 8) if plot_type == "sigma" else (16, 6)
    )  # Increased for better visibility
    fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=False)
    plt.subplots_adjust(wspace=0.02, hspace=0.1)  # Adjusted spacing for colorbars

    # Plot the data in the specified order for each row, skipping noisy and clean
    for i in range(2):
        for j in range(4):
            if j == 0:
                diff = (
                    result_list[0][i + 5][40:-40, 40:-40] - clean[i + 5][40:-40, 40:-40]
                )
                im = axes[i, j].imshow(diff, cmap="coolwarm", vmin=vmin, vmax=vmax)
            elif j == 1:
                diff = (
                    result_list[1][i + 5][40:-40, 40:-40] - clean[i + 5][40:-40, 40:-40]
                )
                im = axes[i, j].imshow(diff, cmap="coolwarm", vmin=vmin, vmax=vmax)
            elif j == 2:
                diff = (
                    result_list[2][i + 5][40:-40, 40:-40] - clean[i + 5][40:-40, 40:-40]
                )
                im = axes[i, j].imshow(diff, cmap="coolwarm", vmin=vmin, vmax=vmax)
            elif j == 3:
                diff = (
                    result_list[3][i + 5][40:-40, 40:-40] - clean[i + 5][40:-40, 40:-40]
                )
                im = axes[i, j].imshow(diff, cmap="coolwarm", vmin=vmin, vmax=vmax)

            # Add colorbar to each difference plot
            fig.colorbar(
                im, ax=axes[i, j], orientation="vertical", fraction=0.046, pad=0.02
            )

            # Remove axis for cleaner view
            axes[i, j].axis("off")

            # Set title only for the first row with smaller font size
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=10)

    # Set the absolute title with Greek letter sigma
    fig.suptitle(
        r"$\sigma = $" + str(noise_sigma), fontsize=16, y=1.05
    )  # Increased font size

    # Set the background color for better visual contrast
    fig.patch.set_facecolor("white")

    # Adjust layout for better spacing and save the figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust the top space for the title

    # Save the figure based on dataset
    if args.dataset == "Heart":
        plt.savefig(
            os.path.join(
                "/home/nadja/tomo_project/Results_Noisier2Inverse_Heart/Plots_Paper",
                f"Plot{noise_sigma}_diff_{plot_type}.svg",
            )
        )
        plt.savefig(
            os.path.join(
                "/home/nadja/tomo_project/Results_Noisier2Inverse_Heart/Plots_Paper",
                f"Plot{noise_sigma}_diff_{plot_type}.png",
            )
        )
    else:
        plt.savefig(
            os.path.join(
                "/home/nadja/tomo_project/Results_Noisier2Inverse/Plots_Paper",
                f"Plot{noise_sigma}_diff_{plot_type}.svg",
            )
        )
        plt.savefig(
            os.path.join(
                "/home/nadja/tomo_project/Results_Noisier2Inverse/Plots_Paper",
                f"Plot{noise_sigma}_diff_{plot_type}.png",
            )
        )


# Titles for each set of plots
titles_z = [
    "difference: ours sobo z - clean",
    "difference: ours z - clean",
    "difference: z - clean",
    "difference: N2I - clean",
]

titles_y = [
    "difference: ours sobo y - clean",
    "difference: ours y - clean",
    "difference: y - clean",
    "difference: N2I - clean",
]

# Create and save z difference plots
create_difference_plots(
    titles_z, [result_sob_z, result_inf_z, result_z, result_n2i], args.noise_sigma, "z"
)
# Create and save y difference plots
create_difference_plots(
    titles_y, [result_sob_y, result_inf_y, result_y, result_n2i], args.noise_sigma, "y"
)
