#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH -N 1
#SBATCH -p standard
#SBATCH --ntasks-per-node=32
#SBATCH --job-name="process_GB"

source /scratch/zt1/project/tiwary-prj/user/ebeyerle/anaconda3/etc/profile.d/conda.sh
#conda activate hoomd


BD=$PWD
DUMP="2.pro.ellipsoid.dump"
rho=0.35
for T in DUMMY; 
do 
		echo $rho $T
		cd ${i}/long/
		cp -v ${BD}/write_energy.sh ./; 
		cp -v ${BD}/print_timesteps.sh ./; 
		cp -v ${BD}/process.py ./
		cp -v ${BD}/C_perp_new_input_features_GB.py ./

		cp -v log.lammps tee.log
		sh write_energy.sh ; 
		sh print_timesteps.sh $DUMP; 
		python process.py ./ $DUMP
		conda activate hoomd
		python C_perp_new_input_features_GB.py ./ $DUMP ${T}
		cd $BD; 
done
