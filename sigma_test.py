# -*- coding: utf-8 -*-
"""
Testing script for Noisier2Inverse
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
    default="/home/nadja/tomo_project/Results_Noisier2Inverse/Model_Weights",
)
parser.add_argument(
    "-noise_sigma",
    "--noise_sigma",
    type=float,
    help="how big is the kernel size of convolution of the NW weights loaded trained NW",
    default=3.0,
)


parser.add_argument(
    "-noise_sigma_test",
    "--noise_sigma_test",
    type=float,
    help="how big is the kernel size of convolution in the data",
    default=3.0,
)
parser.add_argument(
    "-y_z", "--dat", type=str, help="predict on y or on z?", default="z"
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


output_dir = (
    output_dir
    + "/Noise_"
    + args.noise_type
    + "_"
    + str(args.noise_intensity)
    + "_sigma_"
    + str(args.noise_sigma)
    + "_testsigma_"
    + str(args.noise_sigma_test)
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


# Define the Noisier2NoiseRecon class for testing
class Noisier2NoiseReconTest:
    def __init__(self, device: str = "cuda:0"):
        self.net_denoising = UNet(in_channels=1, out_channels=1).to(device)
        self.device = device
        self.angles = args.angles
        self.batch_size = args.batch_size
        self.noise_intensity = args.noise_intensity
        self.noise = args.noise_type
        self.noise_sigma = args.noise_sigma
        self.noise_sigma_test = args.noise_sigma_test

        self.test_dataset = Walnut_test(
            noise_type=self.noise,
            noise_intensity=self.noise_intensity,
            noise_sigma=self.noise_sigma_test,  ### we vary the sigmas here
            data_dir=args.datadir,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )

    def forward(self, reconstruction):
        output_denoising = self.net_denoising(reconstruction.float().to(self.device))
        output_denoising_reco = output_denoising
        output_denoising_sino = self.projection_tomosipo(
            output_denoising, sino=self.angles
        )
        return output_denoising_reco, output_denoising_sino

    def compute_reconstructions(self, sinograms):
        sinograms = sinograms.squeeze(1)
        reconstructions = torch.zeros(
            (sinograms.shape[0], 1, sinograms.shape[-2], sinograms.shape[-2]),
            device=sinograms.device,
        )
        for i in range(sinograms.shape[0]):
            reconstructions[i] = self.fbp_tomosipo(
                sinograms[i].unsqueeze(0).unsqueeze(0)
            )
        return reconstructions

    def projection_tomosipo(self, img, sino):
        angles = sino if isinstance(sino, int) else sino.shape[-1]
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(img.shape[0], img.shape[-1], img.shape[-1]),
            number_of_angles=angles,
            translate=False,
        )
        op = to_autograd(ct.make_operator(geo))
        sino = op((img[:, 0]).to(self.device)).unsqueeze(1)
        return torch.moveaxis(sino, -1, -2)

    def fbp_tomosipo(self, sino):
        angles = sino.shape[-1]
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(sino.shape[0], 336, 336),
            number_of_angles=angles,
            translate=False,
        )
        op = ct.make_operator(geo)
        sino = torch.moveaxis(sino, -1, -2)
        return fbp(op, sino[:, 0]).unsqueeze(1)


# Initialize testing class
N2NR = Noisier2NoiseReconTest(device=device)

weights_path = os.path.join(weights_dir, "emd_model_weights_" + args.dat + ".pth")

if not os.path.exists(weights_path):
    print("File not found, using default weights.")
    weights_path = os.path.join(weights_dir, "emd_model_weights.pth")


print(weights_path, flush=True)
N2NR.net_denoising.load_state_dict(torch.load(weights_path))
N2NR.net_denoising.eval()

# Testing loop
# Save numpy arrays
output_reco_list = []
clean_list = []
noisy_list = []
noisier_list = []
ssim_list = []
psnr_list = []
emd_list = []

# Testing loop
with torch.no_grad():
    all_ssim = []
    all_psnr = []
    all_emd = []

    for batch, data in tqdm(enumerate(N2NR.test_dataloader)):
        clean_test, y_test, z_test = (
            data["clean"].squeeze(),
            data["noisy"].squeeze(),
            data["noisier"].squeeze(),
        )

        # y_test = y_test.unsqueeze(1)
        # y_test = torch.moveaxis(y_test, -1, -2)
        # z_test = z_test.unsqueeze(1)
        # z_test = torch.moveaxis(z_test, -1, -2)
        if len(y_test.shape) < 3:
            y_test = y_test.unsqueeze(0)
            y_test = y_test.unsqueeze(1)
            clean_test = clean_test.unsqueeze(0)

        if len(z_test.shape) < 3:
            z_test = z_test.unsqueeze(0)
            z_test = z_test.unsqueeze(1)

        else:
            y_test = y_test.unsqueeze(1)
            z_test = z_test.unsqueeze(1)

        if "emd_model_weights_y.pth" in weights_path:
            test = y_test.to(device)
            z_test = z_test.to(device)

            recos_test_z = N2NR.compute_reconstructions(z_test)
            recos_test = N2NR.compute_reconstructions(test)
            recos_test_y = torch.clone(recos_test)
        else:
            test = z_test.to(device)
            recos_test_y = N2NR.compute_reconstructions(y_test)
            recos_test = N2NR.compute_reconstructions(test)
            recos_test_z = torch.clone(recos_test)

        output_reco, _ = N2NR.forward(recos_test)
        if "Inference" not in weights_path and args.dat == "z":
            print("we correct")
            "we  have to do the correction only if no inference in the methods name and we predict on z, not on y"
            output_reco = 2 * output_reco - recos_test.to(output_reco.device)
        # Save reconstructions and compute metrics
        print("clean:" + str(clean_test.shape))
        for i in range(len(clean_test)):
            ims_test_np = clean_test[i].cpu().numpy()
            output_reco_np = output_reco[i][0].cpu().numpy()
            recos_test_y_np = recos_test_y[i][0].cpu().numpy()
            recos_test_z_np = recos_test_z[i][0].cpu().numpy()
            # Save each reconstructed image
            output_reco_list.append(output_reco_np)
            clean_list.append(ims_test_np)
            noisy_list.append(recos_test_y_np)
            noisier_list.append(recos_test_z_np)

            plt.imshow(output_reco_np, cmap="gray")
            plt.savefig(
                "/home/nadja/tomo_project/Checks/"
                + "/Noise_"
                + args.noise_type
                + "_"
                + str(args.noise_intensity)
                + "_sigma_"
                + str(args.noise_sigma)
                + "_tchsize_"
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
                + ".png"
            )

            data_range = ims_test_np.max() - ims_test_np.min()

            ssim_value = skm.structural_similarity(
                output_reco_np, ims_test_np, data_range=data_range
            )
            psnr_value = skm.peak_signal_noise_ratio(
                output_reco_np, ims_test_np, data_range=data_range
            )

            noise_test_flattened = np.array(data["noise"][i]).flatten()
            n_flattened = np.array(output_reco[i][0].cpu()).flatten()
            emd_value = wasserstein_distance(noise_test_flattened, n_flattened)

            all_ssim.append(ssim_value)
            all_psnr.append(psnr_value)
            all_emd.append(emd_value)

            ssim_list.append(ssim_value)
            psnr_list.append(psnr_value)
            emd_list.append(emd_value)

    print(f"Average SSIM: {np.mean(all_ssim)}")
    print(f"Average PSNR: {np.mean(all_psnr)}")

# Convert lists to numpy arrays
output_reco_array = np.array(output_reco_list)
clean_array = np.array(clean_list)
recos_test_y_array = np.array(noisy_list)
recos_test_z_array = np.array(noisier_list)


ssim_array = np.array(ssim_list)
psnr_array = np.array(psnr_list)
emd_array = np.array(emd_list)


print(output_reco_array.shape)
print(clean_test.shape, flush=True)
# Save the arrays as .npy files
if "emd_model_weights_y.pth" in weights_path:
    np.savez_compressed(
        os.path.join(output_dir, "output_reco_results_y.npz"),
        output_reco_array=output_reco_array,
        clean_test=clean_array,
        recos_test_z=recos_test_z_array,
        recos_test_y=recos_test_y_array,
    )
    np.save(os.path.join(output_dir, "ssim_y.npy"), ssim_array)
    np.save(os.path.join(output_dir, "psnr_y.npy"), psnr_array)
    np.save(os.path.join(output_dir, "emd_y.npy"), emd_array)


# Save the arrays as .npy files
if "emd_model_weights_z.pth" in weights_path:
    np.savez_compressed(
        os.path.join(output_dir, "output_reco_results_z.npz"),
        output_reco_array=output_reco_array,
        clean_test=clean_array,
        recos_test_z=recos_test_z_array,
        recos_test_y=recos_test_y_array,
    )
    np.save(os.path.join(output_dir, "ssim_z.npy"), ssim_array)
    np.save(os.path.join(output_dir, "psnr_z.npy"), psnr_array)
    np.save(os.path.join(output_dir, "emd_z.npy"), emd_array)


# Save the arrays as .npy files
if "emd_model_weights.pth" in weights_path:
    np.savez_compressed(
        os.path.join(output_dir, "output_reco_results_z.npz"),
        output_reco_array=output_reco_array,
        clean_test=clean_array,
        recos_test_z=recos_test_z_array,
        recos_test_y=recos_test_y_array,
    )
    np.save(os.path.join(output_dir, "ssim_z.npy"), ssim_array)
    np.save(os.path.join(output_dir, "psnr_z.npy"), psnr_array)
    np.save(os.path.join(output_dir, "emd_z.npy"), emd_array)
# %%
