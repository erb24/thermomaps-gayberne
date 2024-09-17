#!/usr/bin/env python
# coding: utf-8

import numpy as np
import subprocess
import sys


DUMP = sys.argv[1]
path = sys.argv[2]

data = np.loadtxt(path + '/unformatted_traj.txt')
NFRS = int(subprocess.getoutput('grep -c "TIMESTEP" ' + DUMP))
N = len(data) // NFRS
print(N, NFRS)

ftraj = np.zeros((NFRS, len(data) // NFRS, 3))
for k in range(NFRS):
	ftraj[k, :, :] = data[k * N:(k + 1) * N, :]


PDB = 'top.pdb'
subprocess.call('rm -rfv ' + path + '/' + PDB, shell=True)
f=open(path + '/' + PDB,"w+")
for k in range(NFRS):
	f.write('MODEL %s\n' % (k))
	#f.write(str(nres)+"\n")
	for i in range(N):
		#if i == nres-1:
		#    f.write('C '+str(10.0*avgx[i,k])+" "+str(10.0*avgy[i,k])+" "+str(10.0*avgz[i,k]))
		#else:
		#    f.write('C '+str(10.0*avgx[i,k])+" "+str(10.0*avgy[i,k])+" "+str(10.0*avgz[i,k])+"\n")
		f.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format("ATOM",i+1,"C"," ","X"," ",i+1," ",ftraj[k,i,0],ftraj[k,i,1],ftraj[k,i,2],1.00,0.00,"C"," ")) 
	f.write('TER\n')
	f.write('ENDMDL\n')
f.close()