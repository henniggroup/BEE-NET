# BEE-NET

**Bootstrapped Ensemble of Equivariant Graph Neural Networks** for predicting the Eliashberg spectral function α²F(ω) and superconducting critical temperature T_c.

📄 **Paper:** [Developing a complete AI-accelerated workflow for superconductor discovery](https://www.nature.com/articles/s41524-026-01964-8)
*npj Computational Materials* (2026)

Jason B. Gibson, Ajinkya C. Hire, Pawan Prakash, Philip M. Dee, Benjamin Geisler, Jung Soo Kim, Zhongwei Li, James J. Hamlin, Gregory R. Stewart, P. J. Hirschfeld & Richard G. Hennig

---

## Overview

BEE-NET is a bootstrapped ensemble of 100 equivariant graph neural networks (e3NN) trained on 5,241 DFT-computed Eliashberg spectral functions, a subset of the ~7,000 electron-phonon calculations of [Cerqueira et al.](https://archive.materialscloud.org/records/3kbt5-r3n56) (see [Data](#data)). Two model variants are provided:

- **CSO (Crystal Structure Only):** Takes only the crystal structure as input. Ideal for large-scale screening.
- **CPD (Coarse Phonon Density of States):** Uses crystal structure + coarse phonon DOS for higher accuracy.

Integrated into a multi-stage AI-accelerated discovery pipeline, BEE-NET screened over 1.3 million candidate structures, two of which (Be₂Hf₂Nb and Be₂HfNb₂) were experimentally synthesized and confirmed as superconductors.

### Key metrics (EMD loss, test set)

| Variant | T_c MAE (K) | T_c R² | True Negative Rate |
|---------|-------------|--------|-------------------|
| CSO     | 1.20        | 0.66   | 0.97              |
| CPD     | 0.87        | 0.79   | 0.991             |

---

## Repository structure

```
BEE-NET/
├── notebooks/          # Train models, make predictions, visualize results
├── workflow/            # Scripts for the screening workflow
├── structures/          # 5,241 CIF files for training/testing
├── indices/             # Train/test split indices and bootstrap indices
├── database.json        # α²F(ω), phonon DOS and DFT targets for all 5,241 entries (not tracked in git)
├── models/              # CSO/ and CPD/ ensembles, 100 checkpoints each (not tracked in git)
├── .gitignore
├── .gitattributes
└── README.md
```

---

## Data

`database.json` contains the **full** dataset (training + test), 5,241 entries. Each entry is an Alexandria compound (`ID` = `agm…`) taken from the electron-phonon calculations of Cerqueira, Sanna & Marques ([Materials Cloud archive](https://archive.materialscloud.org/records/3kbt5-r3n56); `dir_name` gives the archive batch). The paper refers to this source as ~7,000 α²F(ω); 5,241 of them are used here.

| Set | Size | Defined by |
|-----|------|------------|
| Full dataset | 5,241 | `database.json`, `structures/` |
| Test (20%) | 1,049 | `indices/idx_test_full_cpd_id.txt` (Alexandria IDs) |
| Training pool (80%) | 4,192 | all entries not in the test set |
| Bootstrap train / validation | 4,192 draws (~63% unique) / remaining ~37% | `indices/idx_train__cpd_{k}.txt`, `indices/idx_val__cpd_{k}.txt` |

Notes:

- The train/val index files hold **row labels of the DataFrame returned by `pd.read_json('database.json')`** (0–5240), not the `index` column. The `index` column is the CIF filename in `structures/` and diverges from the row label after row 2360, so using it in place of the row label would put test entries into training.
- The test file holds Alexandria `ID`s; use `df.set_index('ID')` before selecting it (as in `Pred_*.ipynb`).
- 200 bootstrap train/val pairs are provided; the released ensembles contain 100 models per variant.
- Every bootstrap train/val pair is disjoint from the test set, and train ∪ val equals the 4,192-entry training pool.
- The released ensembles reproduce the paper's test metrics on these 1,049 test entries (e.g. CPD, EMD: λ R² 0.80 / MAE 0.073; ω_log R² 0.83 / MAE 17.4 K; ω₂ R² 0.91 / MAE 14.9 K).

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/Train_CSO.ipynb` | Train the CSO model ensemble |
| `notebooks/Train_CPD.ipynb` | Train the CPD model ensemble |
| `notebooks/Pred_CSO.ipynb`  | Run predictions with the CSO ensemble and evaluate |
| `notebooks/Pred_CPD.ipynb`  | Run predictions with the CPD ensemble and evaluate |
| `notebooks/plot_confusion.ipynb` | Generate confusion matrices and precision-recall curves. Note: it expects `indices/idx_test_full.txt`, `indices/idx_train_full.txt` and `test_preds/*.json`, which are not included in this repository |

## Workflow

The `workflow/` directory contains the scripts for the high-throughput screening pipeline described in the paper. See `workflow/README.md` for details on each script, including:

- Relaxation of candidate structures with M3GNet
- Formation energy and band gap prediction with MEGNet
- T_c prediction with BEE-NET
- DFT electron-phonon calculations with Quantum ESPRESSO

---

## Prerequisites

- [PyTorch](http://pytorch.org)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [e3nn](https://e3nn.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org)
- [ASE](https://wiki.fysik.dtu.dk/ase/)

### Installation

```bash
conda create --name bee_net python=3.9
conda activate bee_net
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
```

---

## Citation

If you use BEE-NET in your research, please cite:

```bibtex
@article{gibson2026beenet,
  title={Developing a complete AI-accelerated workflow for superconductor discovery},
  author={Gibson, Jason B. and Hire, Ajinkya C. and Prakash, Pawan and Dee, Philip M. and Geisler, Benjamin and Kim, Jung Soo and Li, Zhongwei and Hamlin, James J. and Stewart, Gregory R. and Hirschfeld, P. J. and Hennig, Richard G.},
  journal={npj Computational Materials},
  volume={12},
  pages={95},
  year={2026},
  doi={10.1038/s41524-026-01964-8}
}
```
