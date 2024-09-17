#!/usr/bin/env python
# coding: utf-8


import numpy as np
import freud
import subprocess
from scipy import spatial
from tqdm import tqdm


# local S

T_list = [DUMMY0]

big_x_nop_list = []
big_var_list = []
for rho in [0.35]:
    for T in T_list:
        try:
            print(rho, T)
            path = './T_%s_rho_0.35/' % T
            ftraj = np.load(path + 'processed_traj.npy'); np.save(path + 'sparse_processed_traj.npy', ftraj[::10])
            fquats = np.load(path + 'processed_quats.npy'); np.save(path + 'sparse_processed_quats.npy', fquats[::10])

            NFRS = ftraj.shape[0]
            N = ftraj.shape[1]
            
            spacing = np.linspace(0, 342, 10).astype(int)
            angles = np.zeros((ftraj.shape[0], ftraj.shape[1]))
            cos_theta = np.zeros((ftraj.shape[0], ftraj.shape[1]))
            S_theta = np.zeros(ftraj.shape[0])
            local_S = np.zeros((ftraj.shape[0],ftraj.shape[1]))
            theta = np.linspace(0, np.pi, 11)
            dtheta = theta[1] - theta[0]
            n_samples = len(spacing)


            # get location of particle CoM
            #director_list = []
            x_nop_list = []

            director_list = []

            x = np.array([1, 0, 0])


            for k, frame in tqdm(enumerate(ftraj)):
                loc = np.copy(frame)
                rot = spatial.transform.Rotation.from_quat(np.roll(fquats[k], -1, axis = 1)).as_matrix() #rowan.to_matrix(fquats[k])
                director = x
                #director_list.append(director)
                x_nematic = freud.order.Nematic(director)
                x_nematic.compute(fquats[k])
                x_nop_list.append(x_nematic.order)
                eigvals, eigvecs = np.linalg.eig(x_nematic.nematic_tensor)
                cos_theta[k,:] = np.dot(eigvecs[:,0], np.dot(rot, x).T)
                angles[k, :] = np.arccos(cos_theta[k,:])
                hist, bins = np.histogram(angles[k,:], bins = theta)
                P = hist / hist.sum(); P[P == 0.] = 1 / hist.sum()
                S_theta[k] = -np.sum(np.sin((theta[1:] + theta[:-1]) / 2)[P != 0.] * P[P != 0.] * np.log(P[P != 0.]) * dtheta)
                
                loc = np.copy(frame)
                rot = spatial.transform.Rotation.from_quat(fquats[k]).as_matrix() #rowan.to_matrix(fquats[k])
                director = x
                #director_list.append(director)
                x_nematic = freud.order.Nematic(director)
                x_nematic.compute(fquats[k])

                for i in range(ftraj.shape[1]):
                    nn_list = np.argsort((np.sum((ftraj[k, i, :][None,:] - ftraj[k, :, :]) * (ftraj[k, i, :][None,:] - ftraj[k, :, :]), axis = 1)))[1:11]   
                    x_nematic.compute(fquats[k][nn_list])
                    local_S[k, i] = x_nematic.order
                    
            director_list = np.array(director_list)
            x_nop_List = np.array(x_nop_list)
            big_x_nop_list.append(x_nop_List)
            big_var_list.append(np.var(x_nop_list))

        except ValueError:
            print(rho, T)
            pass
        
big_var_list = np.array(big_var_list)
np.save(path + 'local_S.npy', local_S)
np.save(path + 'cos_theta.npy', cos_theta)
np.save(path + 'theta.npy', angles)
np.save(path + 'S_theta.npy', S_theta)
