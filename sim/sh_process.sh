#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH -N 1
#SBATCH -p standard
#SBATCH --ntasks-per-node=8
#SBATCH --job-name="process_GB"

source /scratch/zt1/project/tiwary-prj/user/ebeyerle/anaconda3/etc/profile.d/conda.sh
#conda activate hoomd


BD=$PWD
DUMP="long.pro.ellipsoid.dump"

for T in 1.95
do
	for rho in 0.35; 
	do 
		echo $i
		cd T_${T}_rho_${rho}/; 
		cp -v ${BD}/write_energy.sh ./; 
		cp -v ${BD}/print_timesteps.sh ./; 
		cp -v ${BD}/process.py ./
		cp -v ${BD}/calc_input.py ./
		cp -v ${BD}/calc_PO.py ./
		cp -v ${BD}/calc_S.py ./
		cp -v ${BD}/process.sh ./
		sbatch process.sh
		cd $BD
	done
done
