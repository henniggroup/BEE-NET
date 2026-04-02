# BETE-NET
This repo contains the models, training data and test predictions for the work described in the paper: [Accelerating superconductor discovery through tempered deep learning of the electron-phonon spectral function](https://arxiv.org/abs/2401.16611)

## Files and directories

1. `relax_dask_m3g.py` uses dask to relax the structures using M3GNET and predicts Ef and Eg using MEGNET.
2. `pred_tc_m3gnet.py` predicts Tc using BEENET.
3. `Pred_CSO.ipynb` also predicts Tc.
4. `get_eah_m3gnet.ipynb` computes Ehull.
5. `get_mp_eah.ipynb` computes Ehull from DFT energies.
6. `make_phdos.ipynb` makes the files for running QE calculations of PhDOS.
7. `get_dos_strict.ipynb` reads the PhDOS and filters materials with imaginary phonons
8. `get_dos_cont.ipynb` is used to restart calculations.
9. `make_elph.ipynb` generates the files for running QE calculations of a2F
10. `get_a2f.ipynb` reads the a2F and computes Tc
11. `get_a2f_cont.ipynb` is used to restart calculations.
