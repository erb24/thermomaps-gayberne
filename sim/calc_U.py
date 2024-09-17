#!/usr/bin/env python
# coding: utf-8


import numpy as np
#import freud
import matplotlib.pyplot as plt
#import subprocess
from scipy import spatial
import os
#import sys
#from tqdm import tqdm
print(os.getcwd())

def quat2rot(quat):
	w = quat[:,0]
	x = quat[:,1]
	y = quat[:,2]
	z = quat[:,3]
	rot = np.array([
		[1 - 2 * y**2 - 2 * z**2, 2 * x * y + 2 * w * z, 2 * x * z - 2 * w * y],
		[2 * x * y - 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z + 2 * w * x],
		[ 2 * x * z + 2 * w * y, 2 * y * z - 2 * w * x, 1 - 2 * x**2 - 2 * y**2]
	])
	return rot

def nop(quats, n):
	dp = np.dot(n, np.matmul(n, quat2rot(quats)))
	S = 0.5 * ((3 * dp * dp) - 1).mean()
	return S

def calc_Q(quats, d = np.array([1, 0, 0])):
	from scipy import spatial
	Q_list = []
	d_list = []
	S_list = []
	for k, quat in enumerate(quats):
		u = np.matmul(spatial.transform.Rotation.from_quat(quat).as_matrix(), d)
		Q = np.zeros((3, 3))
		for i in range(u.shape[0]):
			Q += (1.5 * np.outer(u[i,:], u[i,:])) - 0.5 * np.eye(3)
		_, tmp = np.linalg.eigh(Q / u.shape[0])
		d = tmp[:,-1]
		Q_list.append(Q)
		d_list.append(d)
		S_list.append(_[-1])
	Q_list = np.array(Q_list)
	d_list = np.array(d_list)
	S_list = np.array(S_list)
	return Q_list, d_list, S_list

def rot_from_quat(q, r):
	q_s = q[:,0]
	q_v = q[:,1:].ravel()
	
	r_rotated = (q_s * q_s - np.dot(q_v, q_v)) * r + 2 * q_s * (np.cross(q_v, r)) + 2 * (np.dot(q_v, r)) * q_v
	return r_rotated

boxl = -1.1233261982052916e+01 + 2.1257859853235995e+01

for i in [0.5]:
    directories = []
    directories.append('./T_%s_rho_0.35/' %i)

    U_list = []
    timestep = 0.15
    for counter, d in enumerate(directories):
        print(d)
        directory = d

        try:
            ftraj = np.load('%s/sparse_processed_traj.npy' % (d))
            fquats = np.load('%s/sparse_processed_quats.npy' % (d))
        except FileNotFoundError:
            ftraj = np.load('%s/processed_traj.npy' % (d))
            fquats = np.load('%s/processed_quats.npy' % (d))    
            
        avg_U = np.zeros((fquats.shape[1], fquats.shape[1]))

        NFRS = ftraj.shape[0]
        N = ftraj.shape[1]

        x = np.array([1, 0, 0])

        for k, quat in enumerate(fquats):
            d = x
            u = np.matmul(spatial.transform.Rotation.from_quat(np.roll(fquats[k], -1, axis = 1)).as_matrix(), d)
            U = np.dot(u,u.T)
            avg_U += U
            U_list.append(U)

    avg_U = avg_U / k
    U_list = np.array(U_list)
    np.save('./T_%s_rho_0.35/U_avg.npy' %i, avg_U)
    
    im = plt.imshow(abs(U_list).mean(0), cmap = 'binary')
    plt.clim((0.5,1))
    cbar = plt.colorbar(im)
    cbar.set_label(r'$\widehat{u}_i \cdot \widehat{u}_j$')
    cbar.set_ticks([0.5, 0.75, 1.])
    plt.savefig('./T_%s_rho_0.35/U.pdf' %i, dpi = 300)
    #plt.show()
    plt.close()
