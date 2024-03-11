# CRCNS-dendPLRNN
This repository provides the code for the replication of the CRCNS Challenge A from 2009 using the dendPLRNN architecutre by Durstwitz (2022)

This is a slightly altered implementation of the dendPLRNN by the Durstewitz lab (this is the original repo: https://github.com/DurstewitzLab/dendPLRNN/tree/main/BPTT_TF)

I changed the following things:
- adjusted model so that it takes an input term (in the paper referred to as C*s_t)
- I created ‘experiments’ on my data

1. prepare and scale data, put it in the right folder and add an ‘ubermain.py’ to set hyperparams for model training.

To do this, execute prepare_crcns_training and/or test in BPTT_TF (jupyterNB is also present)

2. train model by executing ‘ubermain.py’ in the new experiments added under BPTT/Experiments/my_exp/

3. extract traces and data plots using the ‘real_data_inference.ipynb’ in BPTT_TF.

# **DendPLRNN: BPTT+TF Training**

## Setup
Install your anaconda distribution of choice, e.g. miniconda via the bash
script ```miniconda.sh```:
```
$ ./miniconda.sh
```
Create the local environment `BPTT_TF`:
```
$ conda env create -f environment.yml
```
Activate the environment and install the package
```
$ conda activate BPTT_TF
(BPTT_TF) $ pip install -e .
```

## Running the code
### <u>Reproducing Table 1</u>
The folder `Experiments` contains ready-to-run examples for reproducing results found in Table 1 of the paper. That is, to run trainings on the Lorenz63 data set using the dendritic PLRNN, run the ubermain.py file in the corresponding directory `Experiments/Table1/Lorenz63`:
```
(BPTT_TF) $ python Experiments/Table1/Lorenz63/ubermain.py
```
In the `ubermain.py` file of each subfolder, adjustments regarding the hyperparameter can be performed (all parser arguments can be found in `main.py`), as well as running multiple parallel runs from different initial paramter configurations, by setting `n_runs = x`, where `x` is the number of runs. Setting `n_cpu = x` will ensure that `x` processes are spawned to handle all the runs.

Running `ubermain.py` will create a `results` folder, in which models and tensorboard information will be stored. To track training or inspect trained models of e.g. the Lorenz63 runs produced by running the template `ubermain.py` file mentioned above using tensorboard, call 
```
(BPTT_TF) $ tensorboard --logdir results/Lorenz63/M22B20tau25T200
```

To evaluate metrics on the test set, call `main_eval.py` with the corresponding results subfolder passed to the parser argument `-p`:
```
(BPTT_TF) $ python main_eval.py -p results/Lorenz63/M22B20tau25T200
```

Finally, the jupyter notebook `example.ipynb` contains code that loads a trained model and plots trajectories, where `model_path` and `data_path` have to be set by the user.

### <u>Reproducing Table S2</u>
Similarly, `Experiments/TableS2` contains experiment setups to reproduce vanilla PLRNN results trained with BPTT+TF (see Table S2 in the paper).

# README from Durstewitz: **Tractable Dendritic RNNs for Reconstructing Nonlinear Dynamical Systems** [ICML 2022 Spotlight]
![alt text for screen readers](images/dendrites.png "Augmenting RNN units with dendrites to increase computational power. Image credit goes to Manuel Brenner & Darshana Kalita.")
## About

This repository provides the code to the paper **Tractable Dendritic RNNs for Reconstructing Nonlinear Dynamical Systems** as accepted at the [ICML 2022](https://icml.cc/Conferences/2022). This work augments RNN units with elements from dendritic computation to increase their computational capabilites in turn allowing for low dimensional RNNs. We apply these models to the field of dynamical systems reconstruction, where low-dimensional representations of the underlying system are very much desired.

The repository is split into two codebases providing different approaches to the estimation of parameters of the dendritic, piecewise linear recurrent neural network (dendPLRNN). The folder `BPTT_TF` contains the codebase using backpropagation through time (BPTT) based training paired with sparse teacher forcing (TF), whereas `VI` embeds the dendPLRNN in a variational inference (VI) framework in the form of a sequential variational autoencoder (SVAE). All code is written in Python using [PyTorch](https://pytorch.org/) as the main deep learning framework.

## Citation
If you find the repository and/or paper helpful for your own research, please consider citing [our work](https://proceedings.mlr.press/v162/brenner22a.html):
```

@InProceedings{pmlr-v162-brenner22a,
  title = 	 {Tractable Dendritic {RNN}s for Reconstructing Nonlinear Dynamical Systems},
  author =       {Brenner, Manuel and Hess, Florian and Mikhaeil, Jonas M and Bereska, Leonard F and Monfared, Zahra and Kuo, Po-Chen and Durstewitz, Daniel},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {2292--2320},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/brenner22a/brenner22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/brenner22a.html},
  abstract = 	 {In many scientific disciplines, we are interested in inferring the nonlinear dynamical system underlying a set of observed time series, a challenging task in the face of chaotic behavior and noise. Previous deep learning approaches toward this goal often suffered from a lack of interpretability and tractability. In particular, the high-dimensional latent spaces often required for a faithful embedding, even when the underlying dynamics lives on a lower-dimensional manifold, can hamper theoretical analysis. Motivated by the emerging principles of dendritic computation, we augment a dynamically interpretable and mathematically tractable piecewise-linear (PL) recurrent neural network (RNN) by a linear spline basis expansion. We show that this approach retains all the theoretically appealing properties of the simple PLRNN, yet boosts its capacity for approximating arbitrary nonlinear dynamical systems in comparatively low dimensions. We employ two frameworks for training the system, one combining BPTT with teacher forcing, and another based on fast and scalable variational inference. We show that the dendritically expanded PLRNN achieves better reconstructions with fewer parameters and dimensions on various dynamical systems benchmarks and compares favorably to other methods, while retaining a tractable and interpretable structure.}
}

```

## Software Versions
* Python 3.9
* PyTorch 1.11 + cudatoolkit v11.3
