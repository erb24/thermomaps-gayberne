#!/bin/bash

last=`grep -n "Step Temp" log.lammps | awk -F : '{ print $1}' | tail -1`
sed 1,${last}d log.lammps | awk '{ print $2 }' > tmp
sed '/time$/,$d' tmp > T.txt
