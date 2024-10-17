#!/bin/bash

# Request an interactive session with specific parameters
#SBATCH --partition=a6000
#SBATCH --nodelist=mp-gpu4-a6000-2
#SBATCH --job-name=ct 




python plot_results.py  --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 5 --noise_intensity 1 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse_Heart/" --datadir '/home/nadja/tomo_project/Data_CT/' 