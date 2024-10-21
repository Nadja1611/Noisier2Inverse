#!/bin/bash

# Request an interactive session with specific parameters
#SBATCH --partition=a6000
#SBATCH --nodelist=mp-gpu4-a6000-4
#SBATCH --job-name=n2i



python recon_n2i.py -l "MSE_image" --angles 512  --noise_type "gauss" --learning_rate 5e-7 --outputdir "/home/nadja/tomo_project/Results_Noise2Inverse/" --datadir '/home/nadja/tomo_project/Data/' --noise_sigma 5 --noise_intensity 5 --batch_size 6 > out_CT.out