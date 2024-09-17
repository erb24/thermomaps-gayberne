import numpy as np

for T in [0.5, 0.6, 0.7, 0.8]:
    print(T)
    ftraj = np.load('T_%s_rho_0.35/processed_traj.npy' % T)
    np.save('T_%s_rho_0.35/sparse_processed_traj.npy' % T, ftraj[::100])
    
    fquats = np.load('T_%s_rho_0.35/processed_quats.npy' % T)
    np.save('T_%s_rho_0.35/sparse_processed_quats.npy' % T, fquats[::100])
