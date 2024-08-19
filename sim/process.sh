#!/bin/bash

BD=$PWD
DUMP="pro.ellipsoid.dump"

for rho in 0.55; 
do 
	for T in 1.0; 
	do 
		echo $rho $T
		cd T_${T}_rho_${rho}; 
		cp -v ${BD}/write_energy.sh ./; 
		cp -v ${BD}/print_timesteps.sh ./; 
		cp -v ${BD}/process.py ./
		cp -v ${BD}/calculate_input_features_GB.py ./
		cp -v log.lammps tee.log
		sh write_energy.sh ; 
		sh print_timesteps.sh $DUMP; 
		python process.py ./ $DUMP
		python C_perp_new_input_features_GB.py ./ $DUMP $T
		cd $BD; 
	done; 
done
