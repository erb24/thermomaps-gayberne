This repo showcases the ability of Thermodynamic Maps to learn the isotropic-nematic phase transitions in a system of Gay-Berne ellipsoids. The notebook supports outlines how to infer behavior at the phase transition using only samples from either side of the phase transition. As indicated by the parent repository of this fork, this project is inspired strongly by the following work:

> Herron *et al.*, "Inferring phase transitions and critical exponents from limited observations with Thermodynamic Maps," 	*PNAS*, **121**, 52, 2024 (https://doi.org/10.1073/pnas.2321971121)

The publication corresponding to this repository has been published in *Phys. Rev. Lett.*:
> Beyerle and Tiwary, "Inferring the Isotropic-nematic Phase Transition with Generative Machine Learning,"  *Phys. Rev. Lett*, **135**, 068102 (https://journals.aps.org/prl/accepted/10.1103/1wdj-ym3s)

To install with GPU support for calculations, remove `torch' from the requirements.txt file, then use the following to install torch with CUDA:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Overall, I have used micromamba as the package manager:
```
micromamba create -n tmgb-gpu
micromamba install pip -c conda-forge
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
micromamba install jupyter -c conda-forge
micromamba install matplotlib -c conda-forge
```

Installation performed May 2024

To actually train and evaluate the peroformance of the model, please use the thermomaps_gb.py file in the `analysis` directory. There is a notebook provided in the base directory of the repository, but this notebook is deprecated and not supported. All the scripts and notebooks required to reproduce the results in the *PRL* paper should be present in the `analysis` directory.
