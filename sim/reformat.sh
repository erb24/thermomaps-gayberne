#!/bin/bash

DUMP=$1
directories="./"
for d in $directories 
do 
	sed -i "s#c_q\[1\]#quatw#g" ${d}/${DUMP}
	sed -i "s#c_q\[2\]#quati#g" ${d}/${DUMP}
	sed -i "s#c_q\[3\]#quatj#g" ${d}/${DUMP}
	sed -i "s#c_q\[4\]#quatk#g" ${d}/${DUMP}
	sh dump2pdb.sh ${d}/${DUMP} ${d}
done

