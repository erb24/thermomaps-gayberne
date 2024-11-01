#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH -N 1
#SBATCH -p standard
#SBATCH --ntasks-per-node=16
#SBATCH --job-name="process_GB"
#SBATCH -A tiwary-prj-paid

source /scratch/zt1/project/tiwary-prj/user/ebeyerle/anaconda3/etc/profile.d/conda.sh
#conda activate hoomd


BD=$PWD
DUMP="long.pro.ellipsoid.dump"

cp -v log.lammps tee.log
sh write_energy.sh ;
sh print_timesteps.sh $DUMP;
python process.py ./ $DUMP
conda activate hoomd
python calc_S.py ./ $DUMP
