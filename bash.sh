#!/bin/bash

# Request an interactive session with specific parameters
#SBATCH --partition=a100
#SBATCH --nodelist=mp-gpu4-a100-2
#SBATCH --job-name=ct 




python method_datadomain_inference_EMD.py -l 'DataDomain_MSE_Inference_EMD' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 3 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' > out_CT.out