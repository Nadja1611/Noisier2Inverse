#!/bin/bash

# Request an interactive session with specific parameters
#SBATCH --partition=a6000
#SBATCH --nodelist=mp-gpu4-a6000-4
#SBATCH --job-name=ct 




python test.py -l 'DataDomain_MSE_EMD' --dat 'y' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 2 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l 'DataDomain_MSE_EMD' --dat 'y' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 3 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l 'DataDomain_MSE_EMD' --dat 'y' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 5 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l "DataDomain_MSE_Inference_EMD_Sobolev" --dat 'y' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 2 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l "DataDomain_MSE_Inference_EMD_Sobolev" --dat 'y' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 3 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l "DataDomain_MSE_Inference_EMD_Sobolev" --dat 'y' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 5 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l "DataDomain_MSE_Inference_EMD" --dat 'y' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 2 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l "DataDomain_MSE_Inference_EMD" --dat 'y' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 3 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l "DataDomain_MSE_Inference_EMD" --dat 'y' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 5 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 




python test.py -l 'DataDomain_MSE_EMD' --dat 'z' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 2 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l 'DataDomain_MSE_EMD' --dat 'z' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 3 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l 'DataDomain_MSE_EMD' --dat 'z' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 5 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l "DataDomain_MSE_Inference_EMD_Sobolev" --dat 'z' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 2 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l "DataDomain_MSE_Inference_EMD_Sobolev" --dat 'z' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 3 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l "DataDomain_MSE_Inference_EMD_Sobolev" --dat 'z' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 5 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l "DataDomain_MSE_Inference_EMD" --dat 'z' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 2 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l "DataDomain_MSE_Inference_EMD" --dat 'z' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 3 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 
python test.py -l "DataDomain_MSE_Inference_EMD" --dat 'z' --angles 512  --noise_type "gauss" --learning_rate 5e-4 --noise_sigma 5 --noise_intensity 5 --batch_size 4 --outputdir "/home/nadja/tomo_project/Results_Noisier2Inverse/" --datadir '/home/nadja/tomo_project/Data/' 