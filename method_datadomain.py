# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:51:34 2024

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

from tqdm import tqdm
from model import *
from torch.optim import lr_scheduler
from itertools import combinations
import LION.CTtools.ct_utils as ct
from ts_algorithms import fbp, tv_min2d
import skimage
import argparse
import gc
from scipy.ndimage import gaussian_filter
from dataset import *
from utils_inverse import create_noisy_sinograms
import psutil

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
    help="number of prosqueuejection angles sinogram",
    default=512,
)
parser.add_argument(
    "-batch_size",
    "--batch_size",
    type=int,
    help="number of prosqueuejection angles sinogram",
    default=16,
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    help="which learning rate should be used",
    default=1e-5,
)
parser.add_argument(
    "-noise_type",
    "--noise_type",
    type=str,
    help="add correlated or uncorrelated noise",
    default="uncorrelated",
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
    default="/home/nadja/nadja/tomo_project/LION/logs",
)
parser.add_argument(
    "-w",
    "--weights_dir",
    type=str,
    help="directory to save model weights",
    default="/home/nadja/tomo_project/LION/Noise2Inverse/Model_Weights",
)

args = parser.parse_args()

""" explanation loss_variant"""
## DataDomain_MSE         - loss is computed in data domain, NW operates in recon domain
## ReconDomain_MSE        - loss is computed in recon domain, NW operates in recon domain

## DataDomain_MSE_Inference           - loss is computed in data domain, NW operates in recon domain, we use inference loss
## ReconDomain_MSE_Inference          - loss is computed in recon domain, NW operates in recon domain, we use inference loss
## DataDomain_MSE_Inference_Sobolev           - Sobolev loss is computed in data domain, NW operates in recon domain, we use inference loss

"""specify weight directory"""
weights_dir = (
    args.weights_dir
    + "/Noise_"
    + args.noise_type
    + "_"
    + str(args.noise_intensity)
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
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

device = "cuda:0"


class Noiser2NoiseRecon:
    def __init__(
        self,
        device: str = "cuda:0",
        folds: int = 1,
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
            num_workers=0,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )

    def forward(self, reconstruction):
        #### the input could be the image or the sinogram, it is just called reconstruction as for most cases it is the reconstructionß
        output_denoising = self.net_denoising(reconstruction.float().to(self.device))
        # if we use the method where network operates in reconstruction domain, we have to project images back again
        output_denoising_reco = output_denoising
        output_denoising_sino = self.projection_tomosipo(
            output_denoising, sino=self.angles
        )  # .to(self.device)

        return output_denoising_reco, output_denoising_sino

    def compute_reconstructions(self, sinograms):
        sinograms = sinograms.squeeze()  # .detach().cpu()

        Reconstructions = torch.zeros(
            (sinograms.shape[0], self.folds, sinograms.shape[-2], sinograms.shape[-2]),
            device=sinograms.device,
        )
        number_of_angles = sinograms.shape[-1]
        projection_indices = np.array([i for i in range(0, number_of_angles)])
        for i in range(sinograms.shape[0]):
            Reconstructions[i] = self.fbp_tomosipo(
                sinograms[i].unsqueeze(0).unsqueeze(0),
                translate=False,
                angle_vector=projection_indices,
                folds=1,
            )
        del sinograms
        gc.collect()
        return Reconstructions

    def projection_tomosipo(self, img, sino, translate=False):
        if isinstance(sino, int) == True:
            angles = sino
        else:
            angles = sino.shape[-1]
        # 0.1: Make geometry:
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(img.shape[0], img.shape[-1], img.shape[-1]),
            number_of_angles=angles,
            translate=False,
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
            translate=False,
            angle_vector=angle_vector,
        )
        op = ct.make_operator(geo)
        sino = torch.moveaxis(sino, -1, -2)

        result = fbp(op, sino[:, 0])
        result = result.unsqueeze(1)
        del (sino, op, angles, geo)

        gc.collect()
        return result


#### generate a new output path where the results are stored!
newpath = (
    r"/home/nadja/tomo_project/LION/Results_Paper_new/Noise_"
    + args.noise_type
    + "_"
    + str(args.noise_intensity)
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
if not os.path.exists(newpath):
    os.makedirs(newpath)


###### specifiy training parameters
N_epochs = 4000
learning_rate = args.learning_rate


###### Choose from 'MSE_image', 'MSE_data', 'Sobolev_data'
N2NR = Noiser2NoiseRecon()
N2NR_optimizer = optim.Adam(N2NR.net_denoising.parameters(), lr=learning_rate)


########################### Now training starts ##############
l2_list = []
all_MSEs = []
# Initialize empty tensors for accumulating mean values
all_ssim_y = torch.tensor([])
all_ssim_z = torch.tensor([])

all_psnr_y = torch.tensor([])
all_psnr_z = torch.tensor([])

for epoch in range(N_epochs):
    running_loss = 0
    running_L2_loss = 0

    # (N2NR.train_dataloader, desc='Epoch {}'.format(epoch)) as tepoch:
    for batch, data in enumerate(N2NR.train_dataloader):
        N2NR.net_denoising.train()
        if args.noise_type == "salt_and_pepper":
            ### if salt and pepper noise is chosen, then the noise is added in the recon domain, and that dataloader reads in the noisy recons
            clean, noisy, noisier = data["clean"], data["noisy"], data["noisier"]
            y = torch.tensor(
                create_noisy_sinograms(np.array(noisy[:].squeeze()), N2NR.angles, 0)
            )
            z = torch.tensor(
                create_noisy_sinograms(np.array(noisier[:].squeeze()), N2NR.angles, 0)
            )
            clean, noisy, noisier = (
                clean,
                noisy.to(N2NR.device),
                noisier.to(N2NR.device),
            )
        else:
            ### if gauss noise is chosen, then the noise is added in the data domain, and that dataloader reads in the noisy sinos
            clean, y, z = (
                data["clean"].squeeze(),
                data["noisy"].squeeze(),
                data["noisier"].squeeze(),
            )
            y = y.unsqueeze(1)
            # y = torch.moveaxis(y, -1,-2)
            z = z.unsqueeze(1)
            # z = torch.moveaxis(z, -1,-2)

        if epoch % 1000 == 0:
            with torch.no_grad():
                plt.subplot(1, 2, 1)
                plt.imshow(z[0][0], cmap="gray")
                plt.subplot(1, 2, 2)
                plt.imshow(y[0][0], cmap="gray")
                plt.savefig(newpath + "/" + "sinograms" + ".png")
                plt.close()

        # generate recos from noisier data, z in the paper is the noisier sinogram
        z_reco = N2NR.compute_reconstructions(z.to(device))
        # generate recos from noisy given data y
        y_reco = N2NR.compute_reconstructions(y.to(device))
        # put reconstructions onto device
        # z_reco = z_recos.to(device)
        # y_reco = y_recos.to(device)
        y = y.to(device)
        z = z.to(device)

        if epoch % 1000 == 0:
            with torch.no_grad():
                plt.subplot(1, 2, 1)
                plt.imshow(z_reco[0][0].detach().cpu(), cmap="gray")
                plt.subplot(1, 2, 2)
                plt.imshow(y_reco[0][0].detach().cpu(), cmap="gray")
                plt.savefig(newpath + "/" + "recon" + ".png")
                plt.close()
        N2NR_optimizer.zero_grad()
        output_reco, output_sino = N2NR.forward(z_reco)
        if epoch != 10 and epoch % 2000 != 0:
            del output_reco
        if args.loss_variant == "DataDomain_MSE_Inference_Sobolev":
            loss = sobolev_norm(output_sino.float(), y.float().detach())

        else:
            loss = torch.nn.functional.mse_loss(output_sino.float(), y.float().detach())

        # compute gradient
        loss.backward()
        N2NR_optimizer.step()
        with torch.no_grad():
            running_loss += loss.item()
    gc.collect()

    if epoch == 10:
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(output_reco.detach().cpu()[0, 0])
        plt.subplot(1, 3, 2)
        plt.imshow(output_sino.detach().cpu()[0, 0])
        plt.subplot(1, 3, 3)
        plt.imshow(z_reco.detach().cpu()[0, 0])
        plt.savefig(newpath + "/dataset.png")
        plt.close()

    if epoch % 2000 == 0:
        with torch.no_grad():
            plt.figure(figsize=(10, 10))
            plt.subplot(221)
            plt.imshow(z_reco[0, 0].detach().cpu())
            plt.colorbar()
            plt.title("z_reco")
            plt.subplot(222)
            plt.imshow(output_reco[0, 0].detach().cpu())
            plt.colorbar()
            plt.title("denoised zreco")
            plt.subplot(223)
            plt.imshow(clean[0].detach().cpu(), aspect="auto")
            plt.colorbar()
            plt.title("clean")
            plt.subplot(224)
            plt.imshow(
                torch.abs(output_sino[0, 0].detach().cpu() - y[0, 0].detach().cpu()),
                aspect="auto",
            )
            plt.savefig(newpath + "/image_" + str(epoch))
            plt.close()
    del (output_sino, clean, z_reco)
    del (z, y, y_reco)

    if epoch % 4 == 0:
        MSEs = []
        ssim_y = []
        ssim_z = []
        ssim_cor = []
        ssim_y_cor = []
        psnr_y = []
        psnr_z = []
        psnr_cor = []
        psnr_y_cor = []

        with torch.no_grad():
            N2NR.net_denoising.eval()

            for batch, data in enumerate(N2NR.test_dataloader):
                #### apply gassian noise a second time to make the sinogram even noisier
                if args.noise_type == "salt_and_pepper":
                    clean_test, noisy_test, noisier_test = (
                        data["clean"],
                        data["noisy"],
                        data["noisier"],
                    )
                    y_test = torch.tensor(
                        create_noisy_sinograms(
                            np.array(noisy_test.squeeze()), N2NR.angles, 0
                        )
                    )
                    z_test = torch.tensor(
                        create_noisy_sinograms(
                            np.array(noisier_test.squeeze()), N2NR.angles, 0
                        )
                    )
                    clean_test, noisy_test, noisier_test = (
                        clean_test.squeeze(),
                        noisy_test.to(N2NR.device),
                        noisier_test.to(N2NR.device),
                    )
                else:
                    clean_test, y_test, z_test = (
                        data["clean"].squeeze(),
                        data["noisy"].squeeze(),
                        data["noisier"].squeeze(),
                    )
                    y_test = y_test.unsqueeze(1)
                    # y_test = torch.moveaxis(y_test, -1,-2)
                    z_test = z_test.unsqueeze(1)
                    # z_test = torch.moveaxis(z_test, -1,-2)

                z_test = z_test.to(device)
                y_test = y_test.to(device)
                z_recos_test = N2NR.compute_reconstructions(z_test).detach()
                y_recos_test = N2NR.compute_reconstructions(y_test).detach()

                """corresponds to the case where the neural network operates on recon domain"""
                output_reco, output_sino = N2NR.forward(z_recos_test.to(N2NR.device))
                """ here, we do the correction only in inference """
                output_reco = 2 * output_reco - z_recos_test
                output_reco_y, _ = N2NR.forward(y_recos_test.to(N2NR.device))

                ### We interpret in the case when @inference method is used, the output as corrected one, as correction is in the loss function

                for i in range(len(clean_test)):
                    # Ensure the tensors are on CPU and converted to numpy arrays
                    ims_test_np = clean_test[i].detach().cpu().numpy()
                    output_reco_y_np = output_reco_y[i][0].detach().cpu().numpy()
                    # we also have a look at the output obtained by directly applying NW to z
                    output_reco_z_np = output_reco[i][0].detach().cpu().numpy()

                    # Calculate the data range for SSIM and PSNR
                    data_range = ims_test_np.max() - ims_test_np.min()

                    # Compute SSIM for y and z
                    ssim_y_value = torch.tensor(
                        skimage.metrics.structural_similarity(
                            output_reco_y_np, ims_test_np, data_range=data_range
                        )
                    ).to(device)
                    ssim_z_value = torch.tensor(
                        skimage.metrics.structural_similarity(
                            output_reco_z_np, ims_test_np, data_range=data_range
                        )
                    ).to(device)

                    # Compute PSNR for y and z
                    psnr_y_value = torch.tensor(
                        skimage.metrics.peak_signal_noise_ratio(
                            output_reco_y_np, ims_test_np, data_range=data_range
                        )
                    ).to(device)
                    psnr_z_value = torch.tensor(
                        skimage.metrics.peak_signal_noise_ratio(
                            output_reco_z_np, ims_test_np, data_range=data_range
                        )
                    ).to(device)

                    # Append the computed values to the respective lists
                    ssim_y.append(ssim_y_value)
                    ssim_z.append(ssim_z_value)
                    psnr_y.append(psnr_y_value)
                    psnr_z.append(psnr_z_value)

            # Calculate mean values and append to the tensors
            all_ssim_y = torch.cat(
                (all_ssim_y, torch.tensor([torch.mean(torch.tensor(ssim_y))]))
            )
            all_ssim_z = torch.cat(
                (all_ssim_z, torch.tensor([torch.mean(torch.tensor(ssim_z))]))
            )

            all_psnr_y = torch.cat(
                (all_psnr_y, torch.tensor([torch.mean(torch.tensor(psnr_y))]))
            )
            all_psnr_z = torch.cat(
                (all_psnr_z, torch.tensor([torch.mean(torch.tensor(psnr_z))]))
            )
            ### get back
            del (ssim_y, ssim_z, psnr_y, psnr_z)

            print(psutil.cpu_percent(), flush=True)

            if epoch % 200 == 0:
                with torch.no_grad():
                    plt.figure(figsize=(10, 10))
                    plt.subplot(221)
                    plt.imshow(z_recos_test[0, 0].detach().cpu())
                    plt.colorbar()
                    plt.title("z_reco")
                    plt.subplot(222)
                    plt.imshow(output_reco[0, 0].detach().cpu())
                    plt.colorbar()
                    plt.title("denoised corr")
                    plt.subplot(223)
                    plt.imshow(clean_test[0].detach().cpu(), aspect="auto")
                    plt.colorbar()
                    plt.title("clean")
                    plt.subplot(224)
                    plt.imshow(output_reco_y[0, 0].detach().cpu(), cmap="gray")
                    plt.colorbar()
                    plt.title("denoised y_reco")
                    plt.savefig(newpath + "/image_val_" + str(epoch))
                    plt.close()
            if epoch % 50 == 0:
                plt.imshow(output_reco[0, 0].detach().cpu(), cmap="gray")
                plt.savefig(newpath + "/results" + str(epoch))
                plt.close()

            """ save model weights if epoch > 200 """
            if epoch > 200:
                if all_ssim_z[-1] > torch.max(all_ssim_z[:-1]):
                    weights_path = os.path.join(
                        weights_dir, f"ssim_model_weights.pth"
                    )
                    torch.save(N2NR.net_denoising.state_dict(), weights_path)
                    print(
                        f"Model weights ssim saved at epoch {epoch} to {weights_path}"
                    )
                if all_psnr_z[-1] > torch.max(all_psnr_z[:-1]):
                    weights_path = os.path.join(
                        weights_dir, f"psnr_model_weights.pth"
                    )
                    torch.save(N2NR.net_denoising.state_dict(), weights_path)
                    print(
                        f"Model weights psnr saved at epoch {epoch} to {weights_path}"
                    )
            #### visualize ssim and psnrs on validation data
            if epoch % 200 == 0:
                plt.figure()
                plt.subplot(121)
                plt.plot(all_ssim_z.detach().cpu(), label="correction, pred on z")
                plt.plot(all_ssim_y.detach().cpu(), label="predict on y")

                plt.legend()
                plt.title("SSIM validation " + args.loss_variant)
                plt.subplot(122)
                plt.plot(all_psnr_y.detach().cpu(), label="predict on y")
                plt.plot(all_psnr_z.detach().cpu(), label="predict on z")
                plt.legend()
                plt.title("PSNR_validation " + args.loss_variant)
                plt.savefig(newpath + "/ssim_psnr_epoch" + str(epoch))
                plt.close()
            del (ims_test_np, output_reco_y_np, output_reco_z_np)
            torch.cuda.empty_cache()

# %%
