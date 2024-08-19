#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/lherron2/thermomaps-ising/blob/main/thermomaps_ising.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


import numpy as np
import sys
from tqdm import tqdm

from ising.observables import Energy, Magnetization
from ising.samplers import SwendsenWangSampler, SingleSpinFlipSampler
from ising.base import IsingModel

from data.trajectory import EnsembleTrajectory, MultiEnsembleTrajectory
from data.dataset import MultiEnsembleDataset
from data.generic import Summary

from tm.core.prior import GlobalEquilibriumHarmonicPrior, UnitNormalPrior
from tm.core.backbone import ConvBackbone
from tm.core.diffusion_model import DiffusionTrainer, SteeredDiffusionSampler
from tm.core.diffusion_process import VPDiffusion
from tm.architectures.unet_2d_mid_attn import Unet2D
from tm.architectures.unet_1d import Unet1D

# extra libraries I am importing to implement the kludge
from data.observables import Observable
from sklearn.model_selection import ShuffleSplit
from tm.core.loader import Loader
MODEL_NAME = sys.argv[1]


import matplotlib.pyplot as plt


trajectories = []
for temperature in [0.2, 0.5, 1.0, 1.2, 1.5, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4]:
    IM = IsingModel(sampler=SingleSpinFlipSampler, size = 8, warmup = 1000, temp = np.round(temperature,1), Jx = 1, Jy = 1)
    tmp = np.load('%sT/orientations.npy' % temperature)[1000:,:320,:]
    IM.trajectory.coordinates = tmp
    IM.trajectory.summary.size = 320
    IM.trajectory.summary.name = 'Gay-Berne'
    #nrg = np.load('%sT/e.npy' % temperature)
    mag = np.zeros(tmp.shape[0])
    for k in range(tmp.shape[0]):
        Q = np.matmul(tmp[k].T, tmp[k]) / tmp.shape[1] - (1/3) * np.eye(3)
        eigvals, _ = np.linalg.eigh(Q)
        mag[k] = 1.5 * np.max(eigvals)    
    #IM.trajectory.observables['energy'] = nrg
    IM.trajectory.observables['magnetization'] = mag
    trajectories.append(IM.trajectory)
dataset = MultiEnsembleDataset(trajectories, Summary())



df = dataset.to_dataframe()


sub_df = df[df['temperature'].isin([1.0,2.0])]
sub_dataset = dataset.from_dataframe(sub_df)


simulated_M_v_T_mean = {t.state_variables['temperature']: t.observables['magnetization'].mean() for t in dataset.trajectories}
#simulated_E_v_T_mean = {t.state_variables['temperature']: t.observables['energy'].mean() for t in dataset.trajectories}
simulated_M_v_T_std = {t.state_variables['temperature']: t.observables['magnetization'].std() for t in dataset.trajectories}
#simulated_E_v_T_std = {t.state_variables['temperature']: t.observables['energy'].std() for t in dataset.trajectories}

train_M_v_T = {traj.state_variables['temperature']: simulated_M_v_T_mean[traj.state_variables['temperature']] for traj in sub_dataset.trajectories}
#train_E_v_T = {traj.state_variables['temperature']: simulated_E_v_T_mean[traj.state_variables['temperature']] for traj in sub_dataset.trajectories}


orientations_list = []
for temperature in [1.0, 2.0]: #[0.2, 0.5, 1.0, 1.2, 1.5, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4]:
    orientations_list.append(np.load('%sT/orientations.npy' % temperature)[1000:,:320,:])
orientations = np.concatenate(orientations_list)


data = np.zeros((orientations.shape[0], orientations.shape[1], orientations.shape[2] + 1))
data[:,:,:3] = orientations
temperatures = np.array([1.0, 2.0]) #np.array([0.2, 0.5, 1.0, 1.2, 1.5, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4])
T_list = np.zeros(data.shape[0])
for i in range(len(temperatures)):
    T_list[len(T_list) // len(temperatures) * (i):len(T_list) // len(temperatures) * ((i+1))] = temperatures[i]

data[:,:,-1] += T_list[:,None]

data = data.transpose(0, 2, 1)

# Verify the new shape
print(data.shape)

trajectory={'coordinate':[],'state_variables':[]}

trajectory['coordinate'] = data #orientations.transpose(0, 2, 1)
trajectory['state_variables'] = T_list


train_loader = Loader(data=trajectory['coordinate'], temperatures=trajectory['state_variables'][:, None], control_dims=(3,4))#, **TMLoader_kwargs)
print(train_loader.num_dims, train_loader.num_channels)


prior = GlobalEquilibriumHarmonicPrior(shape=train_loader.data.shape, channels_info={"coordinate": [0,1,2], "fluctuation": [3]})


model = Unet1D(dim=16, dim_mults=(1, 2, 4, 8), resnet_block_groups=8, channels=4)


backbone = ConvBackbone(model=model,
                        data_shape=train_loader.data_dim,
                        target_shape=320,
                        num_dims=3,
                        lr=DUMMY0,
                        eval_mode="train",
                       interpolate=False)


diffusion = VPDiffusion(num_diffusion_timesteps=100)


trainer = DiffusionTrainer(diffusion,
                           backbone,
                           train_loader,
                           prior,
                           model_dir="./models/", # save models every epoch
                           pred_type="noise" #, # set to "noise" or "x0"
                           #test_loader=test_loader # optional
                           )


trainer.train(NSTEP, loss_type="smooth_l1", batch_size=DUMMY1)
# Note that the test loss is usually slightly lower than the training loss. This is because
# the training loss is averaged over each epoch (which contains many updates to the weights
# via backprop) while the test loss is evaluated at the end of each epoch. Is there a
# better way to do this? Probably. But it's low priority at the moment.


sampler = SteeredDiffusionSampler(diffusion,
                                  backbone,
                                  train_loader,
                                  prior,
                                  pred_type='noise', # must be the same as in DiffusionTrainer
                                  )


backbone.save_state(MODEL_NAME, NSTEP)


trajectories = []
pbar = tqdm([0.2, 0.5, 1.0, 1.2, 1.5, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4])
for temperature in pbar:
  pbar.set_description(f"Generating at T={temperature:.1f}")
  samples = sampler.sample_loop(num_samples=10000, batch_size=10000, temperature=temperature)
  coords = samples[:,:-1,:].numpy() # take coordinate dimension
  # normalize
  coords = coords / np.linalg.norm(coords, axis = 1)[:, None,:]

  # store in trajectory
  trajectory = EnsembleTrajectory(summary=Summary(info="Generated trajectory"),
                                  state_variables=Summary(temperature=temperature),
                                  coordinates=coords)

  mag = np.zeros(coords.shape[0])
  # evaluate observables over trajectory coordinates and add to trajectory object
  for k in range(coords.shape[0]):
      Q = np.matmul(coords[k], coords[k].T) / coords.shape[2] - (1/3) * np.eye(3)
      eigvals, _ = np.linalg.eigh(Q)
      mag[k] = 1.5 * np.max(eigvals)

  test = Magnetization()
  test.set(mag)
  trajectory.add_observable(test)
  trajectories.append(trajectory)

np.save(MODEL_NAME + 'magnetization_redo_short.npy', np.array([(t.state_variables['temperature'], t.observables['magnetization'].quantity.mean(), t.observables['magnetization'].quantity.std()) for t in trajectories]))
np.save(MODEL_NAME + '_samples_redo_short.npy', np.array([(t.observables['magnetization'].quantity) for t in trajectories]))
