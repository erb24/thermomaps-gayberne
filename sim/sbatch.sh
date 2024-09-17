#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node=16
#SBATCH --job-name="nvt-ASPHERE"

ITEMP=1.69
FTEMP=1.5

date
module load LAMMPS/14Dec21-intel
module load intel
module list

lmp < in.equil

mv -v log.lammps equil.log.lammps

sed "s#ITEMP#${ITEMP}#g" in.tmp > in.ellipsoid
sed -i "s#FTEMP#${FTEMP}#g" in.ellipsoid

lmp < in.ellipsoid

date
