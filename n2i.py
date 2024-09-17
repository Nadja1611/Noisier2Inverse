# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:51:34 2024

@author: nadja <3 Johnny
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
    "-noise_intensity",
    "--noise_intensity",
    type=float,
    help="how intense should salt and pepper noise be",
    default=0.05,
)
parser.add_argument(
    "-o",
    "--logdir",
    type=str,
    help="directory for log files",
    default="/home/nadja/tomo_project/LION/logs",
)
parser.add_argument(
    "-w",
    "--weights_dir",
    type=str,
    help="directory to save model weights",
    default="/home/nadja/tomo_project/LION/Noise2Inverse/Model_Weights/",
)

args = parser.parse_args()
writer = SummaryWriter(log_dir=args.logdir)

""" explanation loss_variant"""
## DataDomain_MSE         - loss is computed in data domain, NW operates in recon domain (sparse2inverse)
## ReconDomain_MSE        - loss is computed in recon domain, NW operates in recon domain
## DataDomain_Sobolev     - Sobolev loss is computed in data domain, NW operates in recon domain

weights_dir = (
    args.weights_dir
    + "N2I_"
    + args.noise_type
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

        # Dataset
        self.train_dataset = Walnut(
            noise_type=self.noise, noise_intensity=self.noise_intensity, train=True
        )
        self.test_dataset = Walnut(
            noise_type=self.noise, noise_intensity=self.noise_intensity, train=False
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
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
        theta = np.linspace(0.0, 180.0, number_of_angles, endpoint=False)
        angles_four_folds = theta[indices_four_folds]
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
    r"/home/nadja/tomo_project/LION/Results_September/N2I_"
    + args.noise_type
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
N_epochs = 3000
learning_rate = args.learning_rate



###### Choose from 'MSE_image', 'MSE_data', 'Sobolev_data'
N2I = Noise2Inverse()
N2I_optimizer = optim.Adam(N2I.net_denoising.parameters(), lr=learning_rate)


########################### Now training starts ##############
l2_list = []
all_MSEs = []
# Initialize empty tensors for accumulating mean values
all_ssim = torch.tensor([])
all_psnr = torch.tensor([])

# Initialize old ssim and psnr
old_ssim = 0.1
old_psnr = 0.1

for epoch in range(N_epochs):
    running_loss = 0
    running_L2_loss = 0

    # (N2I.train_dataloader, desc='Epoch {}'.format(epoch)) as tepoch:
    for batch, data in enumerate(N2I.train_dataloader):
        N2I.net_denoising.train()
        N2I_optimizer.zero_grad()

        clean, noisy = data["clean"].squeeze(), data["noisy"].squeeze()
        y = noisy.unsqueeze(1).to(N2I.device)
        # y = torch.moveaxis(y, -1,-2)

        # generate recos from noisy given data y
        y_recos, indices = N2I.prepare_batch(y.to(device))

        rand_ints = np.random.permutation(N2I.folds)
        loss_indices = indices[rand_ints[N2I.folds - 1]]
        input_x_den = torch.mean(y_recos[:, rand_ints[: N2I.folds - 1]], axis=1)
        input_x_den = input_x_den.unsqueeze(1)
        # target is \tilde{x}_J, as |J| = 1 , no mean required
        target = y_recos[:, rand_ints[N2I.folds - 1]]
        target = target.unsqueeze(1).to(N2I.device)
        # divide by 5 for faster NW convergence
        input_x_den = input_x_den / 5
        output_reco, output_sino = N2I.forward(input_x_den.to(N2I.device), y)

        target = target / 5

        output_sino *= 5
        if args.loss_variant == "MSE_image":
            loss = torch.nn.functional.mse_loss(output_reco.float(), target.float())
            with torch.no_grad():
                l2_loss = torch.nn.functional.mse_loss(
                    output_reco.float().squeeze(), clean.float().to(device)
                )
        elif args.loss_variant == "MSE_data":
            loss = torch.nn.functional.mse_loss(
                output_sino[:, :, :, loss_indices].float(),
                y[:, :, :, loss_indices].float().to(N2I.device),
            )
            with torch.no_grad():
                l2_loss = torch.nn.functional.mse_loss(
                    output_reco.float().squeeze(), clean.float().to(device)
                )

        loss.backward()
        N2I_optimizer.step()
        running_loss += loss.item()
        running_L2_loss += l2_loss.item()
        l2_list.append(running_L2_loss)
        # put reconstructions onto device

        print("we have epoch " + str(epoch))

        if epoch % 100 == 0:
            plt.subplot(1, 2, 1)
            plt.imshow(y[0][0].detach().cpu())
            plt.colorbar()
            plt.title("sino")

            plt.subplot(1, 2, 2)
            plt.imshow(y_recos[0][0])
            plt.title("y reco (target)")
            plt.colorbar()

            plt.savefig(newpath + "/training_data.png")
            plt.close()

        running_loss += loss.item()
        # compute gradient

    # 2_list.append(running_loss/(len(Data_loader)*N2I.batch_size) )
    # np.savez_compressed(newpath+"/l2.npz",l2 = l2_list)

    del (output_reco, output_sino, clean, y_recos)
    gc.collect()

    if epoch % 2 == 0:
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
                    output_reco, output_sino = N2I.forward(
                        input_x_den.to(N2I.device), y_test
                    )
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

            # Calculate mean values and append to the tensors
            all_ssim = torch.cat(
                (all_ssim, torch.tensor([torch.mean(torch.tensor(ssim_y))]))
            )
            all_psnr = torch.cat(
                (all_psnr, torch.tensor([torch.mean(torch.tensor(psnr_y))]))
            )
            """ save model weights if epoch > 1000 """
            if epoch > 10:
                if all_ssim[-1] > old_ssim:
                    old_ssim = all_ssim[-1]
                    weights_path = os.path.join(
                        weights_dir, f"ssim_model_weights_ssim.pth"
                    )
                    torch.save(N2I.net_denoising.state_dict(), weights_path)
                if all_psnr[-1] > old_psnr:
                    old_psnr = all_psnr[-1]
                    weights_path = os.path.join(
                        weights_dir, f"psnr_model_weights.pth"
                    )
                    torch.save(N2I.net_denoising.state_dict(), weights_path)
                    print(f"Model weights psr saved at epoch {epoch} to {weights_path}")

            if epoch % 100 == 0:
                with torch.no_grad():
                    plt.figure(figsize=(10, 10))
                    plt.subplot(221)
                    plt.imshow(y_recos_test[0, 0].detach().cpu())
                    plt.colorbar()
                    plt.title("y_reco")
                    plt.subplot(222)
                    plt.imshow(output_reco[0, 0].detach().cpu())
                    plt.colorbar()
                    plt.title("denoised corr")
                    plt.subplot(223)
                    plt.imshow(clean_test[0].detach().cpu(), aspect="auto")
                    plt.colorbar()
                    plt.title("clean")
                    plt.subplot(224)
                    plt.imshow(final_recos[0, 0].detach().cpu(), cmap="gray")
                    plt.colorbar()
                    plt.title("denoised y_reco")
                    plt.savefig(newpath + "/image_val_" + str(epoch))
                    plt.close()
            if epoch % 100 == 0:
                plt.imshow(final_recos[0, 0].detach().cpu(), cmap="gray")
                plt.savefig(newpath + "/results" + str(epoch))
                plt.close()

        # if epoch>30:

        # np.savez_compressed(newpath + "/Result_ssim.npz", output_reco.detach().cpu(), ssim = all_ssim)

        # np.savez_compressed(newpath + "/Result_psnr.npz", output_reco.detach().cpu(), psnr = all_psnr)

        #### visualize ssim and psnrs on validation data
        if epoch % 100 == 0:
            plt.figure()
            plt.subplot(121)
            plt.plot(all_ssim.detach().cpu(), label="ssim")
            plt.legend()
            plt.title("SSIM validation " + args.loss_variant)
            plt.subplot(122)
            plt.plot(all_psnr.detach().cpu(), label="psnr")
            plt.legend()
            plt.title("PSNR_validation " + args.loss_variant)
            plt.savefig(newpath + "/ssim_psnr_epoch" + str(epoch))
            plt.close()
            gc.collect()
# %%
