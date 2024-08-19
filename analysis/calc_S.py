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



directories = []
directories.append(sys.argv[1]); print(directories)

timestep = 0.15
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

		x_nop_list = np.array(x_nop_list)
		np.save(directory + '/S.npy', x_nop_list)
	except ValueError:

		pass
