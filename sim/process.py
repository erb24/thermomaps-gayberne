#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys


PATH = sys.argv[1]; print(PATH)



traj = []
quats = []
N = 343
NF = 12
counter = 0
with open(PATH + 'pro.ellipsoid.dump', "r") as f:
    for line in tqdm(f):
        tmp = line.split()
        if (len(tmp) == NF):
            #print(tmp)
            tmp_quats = [[float(tmp[5]), float(tmp[6]), float(tmp[7]), float(tmp[8])]]
            tmp = [[float(tmp[2]), float(tmp[3]), float(tmp[4])]]
            for i in range(N):
                dummy = f.readline()
                if len(dummy.split()) != NF:
                    break
                else:
                    tmp2 = dummy.split()
                    tmp.append([float(tmp2[2]), float(tmp2[3]), float(tmp2[4])])
                    tmp_quats.append([float(tmp2[5]), float(tmp2[6]), float(tmp2[7]), float(tmp2[8])])
            if len(tmp) == N:
                traj.append(tmp)
                quats.append(tmp_quats)


ftraj = np.zeros((len(traj), np.array(traj[0]).shape[0], np.array(traj[0]).shape[1]))
fquats = np.zeros((len(quats), np.array(quats[0]).shape[0], np.array(quats[0]).shape[1]))

for n, itraj in enumerate(traj):
    ftraj[n,:,:] = np.array(itraj)
    fquats[n,:,:] = np.array(quats[n])


timesteps = np.loadtxt(PATH + 'timesteps.txt')

unique_indices = np.unique(timesteps)
reverse_unique_indices = unique_indices[::-1]
reverse_timesteps = timesteps[::-1]
reverse_ftraj = ftraj[::-1,:,:]
reverse_fquats = fquats[::-1,:,:]

unique_traj = np.zeros((len(unique_indices), ftraj.shape[1], ftraj.shape[2]))
unique_quats = np.zeros((len(unique_indices), fquats.shape[1], fquats.shape[2]))
step_list = np.zeros(unique_traj.shape[0], dtype = int)
index_list = np.zeros(unique_traj.shape[0], dtype = int)

counter = 0
skip_counter = 0
for i, k in tqdm(enumerate(reverse_timesteps)):
    if k in step_list:
        skip_counter += 1
        print(skip_counter)
    else:
        unique_traj[counter,:,:] = reverse_ftraj[i,:,:]
        unique_quats[counter,:,:] = reverse_fquats[i,:,:]
        index_list[counter] = i
        step_list[counter] = k
        counter += 1
        
unique_traj = unique_traj[::-1,:,:]
unique_quats = unique_quats[::-1,:,:]
index_list = np.array(index_list, dtype = int)


'''timestep = 0.015
with open(PATH + 'COLVAR') as f:
    tmp = []
    for line in f:
        if line[0] == '#':
            pass
        else:
            if len(line.split()) == 6:
                tmp.append(line.split())
            else:
                pass
colvar = np.array(tmp, dtype = float)[::-1]
tmp = np.copy(colvar)
dummy = []
dummy.append(colvar[0,:])
counter = 0
for k in range(1, len(colvar)):
    if np.allclose(colvar[k,0] - dummy[counter][0], -timestep):
        dummy.append(colvar[k,:])
        counter += 1
colvar = np.array(dummy)[::-1]'''


timestep = 100
with open(PATH + 'thermo.txt') as f:
    tmp = []
    for line in f:
        if line[0] == '#':
            pass
        else:
            if len(line.split()) == 9:
                tmp.append(line.split())
            else:
                pass
thermo = np.array(tmp, dtype = float)[::-1]
tmp = np.copy(thermo)
dummy = []
dummy.append(thermo[0,:])
counter = 0
for k in range(1, len(thermo)):
    if np.allclose(thermo[k,0] - dummy[counter][0], -timestep):
        dummy.append(thermo[k,:])
        counter += 1
thermo = np.array(dummy)[::-1]


np.save(PATH + 'processed_traj.npy', unique_traj[:len(thermo),:,:])
np.save(PATH + 'processed_quats.npy', unique_quats[:len(thermo),:,:])
#np.save(PATH + 'colvar.npy', colvar[:len(thermo),:])
np.savetxt(PATH + 'e.txt', thermo[:,4])
