#!/usr/bin/env python
# coding: utf-8


import numpy as np
import freud
import matplotlib.pyplot as plt
import subprocess
from scipy import spatial
import os
import sys
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


# coordination number for molecules perpendicular to the director axis
def C_perp_new(ftraj, n, boxl, r_cut = 3.0):
	N = ftraj.shape[1]
	C = np.zeros(ftraj.shape[0])
	for i in range(N):
		#if i % 10 == 0: print(i)
		for j in range(i,N):
			if i == j: pass
			tmp = ftraj[:,i,:] - ftraj[:,j,:]
			# pbcs
			tmp = tmp - boxl * np.trunc(2 * tmp/boxl)
			rij = np.linalg.norm(tmp - n * (tmp * n).sum(1)[:,None], axis = 1)
			C += (1 - (rij / r_cut)**6) / (1 - (rij / r_cut)**12)
	return C / N 

# coordination number for molecules parallel to the director axis
def C_parallel(ftraj, n, boxl, r_cut = 3.0):
	N = ftraj.shape[1]
	C = np.zeros(ftraj.shape[0])
	for i in range(N):
		#if i % 10 == 0: print(i)
		for j in range(i,N):
			if i == j: pass
			else:
				tmp = ftraj[:,i,:] - ftraj[:,j,:]
				# pbcs
				tmp = tmp - boxl * np.trunc(2 * tmp/boxl)
				rij = abs(((tmp) * n).sum(1))
				C += (1 - (rij / r_cut)**6) / (1 - (rij / r_cut)**12)
	return C / N 

# PCA for a biased trajectory

def biased_PCA(data, weights):
	whitened_data = (data - data.mean(0)) / data.std(0)
	weighted_data = data * weights[:, None] * np.sqrt((weights[:, None] / weights.sum()))
	covar = np.matmul(weighted_data.T, weighted_data)
	eigvals, eigvecs = np.linalg.eigh(covar)
	PCs = np.matmul(data, eigvecs)
	return covar, eigvals, eigvecs, PCs


directories = []
directories.append(sys.argv[1]); print(directories)


timestep = 0.15
big_x_nop_list = []
big_director_list = []
big_energy_list = []
big_q4_list = []
big_q6_list = []
for counter, d in enumerate(directories):
	print(d)
	directory = d
	dummy = subprocess.getoutput("head %s/%s | grep -m 1 -A 1 'BOX' | sed -n '2p'" % (directories[0], sys.argv[2])).split()
	L_min = float(dummy[0])
	L_max = float(dummy[1])
	L = L_max - L_min
	box = freud.box.Box.cube(L)
	boxl = L_max - L_min
	
	try:
		ftraj = np.load('%s/processed_traj.npy' % (d))
		fquats = np.load('%s/processed_quats.npy' % (d))
		nrg = np.loadtxt('%s/e.txt' % (d))

		NFRS = ftraj.shape[0]
		N = ftraj.shape[1]


		# get location of particle CoM
		#director_list = []
		x_nop_list = []

		director_list = []

		x = np.array([1, 0, 0])


		for k, frame in enumerate(ftraj):
			loc = np.copy(frame)
			director = x
			#director_list.append(director)
			x_nematic = freud.order.Nematic(director)
			x_nematic.compute(fquats[k])
			x_nop_list.append(x_nematic.order)
			director_list.append(x_nematic.director)

		director_list = np.array(director_list)
		x_nop_list = np.array(x_nop_list)
		
		big_x_nop_list.append(x_nop_list)
		big_director_list.append(director_list)
		big_energy_list.append(nrg)

		
		box_traj = ftraj - L_min - (L_max / 2)
		q4_list = np.zeros(len(box_traj))
		q6_list = np.zeros(len(box_traj))
		q4 = freud.order.Steinhardt(l = 4, average = True)
		q6 = freud.order.Steinhardt(l = 6, average = True)
		for k in range(box_traj.shape[0]):
			if k % 1000 == 0: print(k)
			q4.compute((box, box_traj[k,:,:]), {'r_max':3})
			q4_list[k] = q4.order

			q6.compute((box, box_traj[k,:,:]), {'r_max':3})
			q6_list[k] = q6.order
			
		
		big_q4_list.append(q4_list)
		big_q6_list.append(q6_list)
		
		pi = np.pi
		PO_list = np.zeros(len(box_traj[:]))
		for k, quat in enumerate(fquats[:]):
			PO = 0.0
			if k % 1000 == 0: print(k)
			coord = ftraj[k]
			d = director_list[k]
			for n in range(quat.shape[0]):
				u = rot_from_quat(quat[n][None,:], x)
				cost = np.dot(u, d)
				PO += (np.exp((1.j * 2 * pi * np.dot(coord[n], d)) / 3.))
			PO = abs(PO) / quat.shape[0]

			PO_list[k] = PO 
			
		
		colvars = np.column_stack([x_nop_list, PO_list, q4_list, q6_list])
		
		np.save(directory + '/input_features.npy', colvars)
		if counter == 0:
			big_traj = ftraj
		else:
			big_traj = np.concatenate((big_traj, ftraj))
	except ValueError:

		pass
