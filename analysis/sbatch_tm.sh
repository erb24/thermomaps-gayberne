#!/bin/bash

for nstep in 25 50 100 250
do
	for bs in 128 256 512 1024 2048
	do
		for lr in 0.001 0.005 0.01 0.05 0.1
		do

			sed "s#NSTEP#${nstep}#g" thermomaps_gb.py > tmp
			sed "s#DUMMY1#${bs}#g" tmp > tmp2
			sed "s#DUMMY0#${lr}#g" tmp2 > nstep_${nstep}_bs_${bs}_lr_${lr}_tm.py
			echo "python nstep_${nstep}_bs_${bs}_lr_${lr}_tm.py nstep_${nstep}_bs_${bs}_lr_${lr}" > tmp3
			cat dummy.sh tmp3 > run_nstep_${nstep}_bs_${bs}_lr_${lr}_tm.sh
			#echo "" >> run_nstep_${nstep}_bs_${bs}_lr_${lr}_tm.sh
			sbatch run_nstep_${nstep}_bs_${bs}_lr_${lr}_tm.sh $nstep $bs $lr
		done
	done
done

rm -rfv tmp tmp[1-3]
