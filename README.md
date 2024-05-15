This repo showcases the ability of Thermodynamic Maps to learn the phase transitions in a system of Gay-Berne ellipsoids. The notebook supports outlines how to infer behavior at the phase transition using only samples from either side of the phase transition.

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
