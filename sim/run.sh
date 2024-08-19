#!/bin/bash
#SBATCH -t 2-0:00:00
#SBATCH -N 1
#SBATCH -p standard
#SBATCH --ntasks-per-node=16
#SBATCH --job-name="nvt-ASPHERE"

date
module load lammps/20210310
module list

#lmp < in.equil
#lmp < in.ellipsoid | tee tee.log
lmp < in.restart | tee -a tee.log
date
