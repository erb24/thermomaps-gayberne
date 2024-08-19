#!/bin/bash

BD=$PWD

for rho in 0.35 0.4; 
do 
	for T in `seq 0.05 0.05 0.95`; 
	do 
		echo $rho $T
		cd rho_${rho}/T_${T}/; 
		cp -v ${BD}/write_energy.sh ./; 
		cp -v ${BD}/print_timesteps.sh ./; 
		cp -v ${BD}/process.py ./
		sh write_energy.sh ; 
		sh print_timesteps.sh pro2.ellipsoid.dump; 
		python process.py ./
		cd $BD; 
	done; 
done
