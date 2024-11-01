#!/usr/bin/env python
# coding: utf-8


import numpy as np


tm_orientations = np.transpose(np.load('nstep_100_bs_512_lr_0.01_coords_redo_short.npy'), (0,1,3,2))


orientations_list = []
e_list = []
T_list = [DUMMY]
for i, T in enumerate(T_list):
    orientations_list.append(np.load('%sT/orientations.npy' % T)[1000:,:320,:])
    e_list.append(np.load('%sT/e.npy' % T)[1000:10000])
    print(len(orientations_list[i]), len(e_list[i]))
orientations = np.array(orientations_list)
U = np.array(e_list)
print(orientations.shape, U.shape)


n = 0
d_list = np.zeros((orientations[n].shape[0], tm_orientations[n].shape[0]))
for k2 in range(tm_orientations[n].shape[0]):
    if k2 % 100 == 0: print(k2)
    for k1 in range(orientations[n].shape[0]): 
        d_list[k1, k2] = np.trace(abs(np.dot(orientations[n][k1,:,:], tm_orientations[n][k2,:,:].T)))


np.save('%T/dist_list.npy' % T, d_list)