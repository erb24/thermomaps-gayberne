#!/bin/bash

XYZ=$1
FOLDER=$2
awk 'NF==12 {print $0 }' $XYZ > tmp
awk '{ print $3" "$4" "$5 }' tmp > ${FOLDER}/unformatted_traj.txt
awk '{ print $6" "$7" "$8" "$9 }' tmp > ${FOLDER}/quats
rm -rfv tmp

python dump_PDB.py $XYZ $FOLDER
