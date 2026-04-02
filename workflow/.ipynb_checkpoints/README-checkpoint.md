# BETE-NET
This repo contains the models, training data and test predictions for the work described in the paper: [Accelerating superconductor discovery through tempered deep learning of the electron-phonon spectral function](https://arxiv.org/abs/2401.16611)

## Files and directories

The `CSO/` `CPD/` directories contain the models trained with the EMD loss function.

The `workflow/` directory contains the scripts for running the pilot workflow.

The `train/database.json` file contains the database used to train the model.

The `train/structures/` directory contains the corresponding cif files.

The `train/indices/` directory contain the training `indices/idx_train_full.txt`, and testing `indices/idx_test_full.txt` indices as well as the indices used for bootstrapping.

The `notebook/` directory contains notebooks to visualize the predictions, train the models and make predictions on a dataset.

##  Prerequisites

This package requires:


- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org)

If you are new to Python, the easiest way of installing the prerequisites is via [conda](https://conda.io/docs/index.html). After installing [conda](http://conda.pydata.org/), run the following command to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) named `bete_net` and install all prerequisites:

```bash
conda create --name bete_net python=3.9
conda activate bete_net
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
```

