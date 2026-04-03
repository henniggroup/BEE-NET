# BEE-NET

**Bootstrapped Ensemble of Equivariant Graph Neural Networks** for predicting the Eliashberg spectral function α²F(ω) and superconducting critical temperature T_c.

📄 **Paper:** [Developing a complete AI-accelerated workflow for superconductor discovery](https://www.nature.com/articles/s41524-026-01964-8)
*npj Computational Materials* (2026)

Jason B. Gibson, Ajinkya C. Hire, Pawan Prakash, Philip M. Dee, Benjamin Geisler, Jung Soo Kim, Zhongwei Li, James J. Hamlin, Gregory R. Stewart, P. J. Hirschfeld & Richard G. Hennig

---

## Overview

BEE-NET is a bootstrapped ensemble of 100 equivariant graph neural networks (e3NN) trained on ~7,000 DFT-computed Eliashberg spectral functions. Two model variants are provided:

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
├── .gitignore
├── .gitattributes
└── README.md
```

## Model weights & large files (Hugging Face)

The trained model weights and training database are hosted on Hugging Face due to their size (~12 GB total):

🤗 **[huggingface.co/paprakash/BEE-NET](https://huggingface.co/paprakash/BEE-NET)**

| File/Folder | Description |
|-------------|-------------|
| `CSO/`      | 100 CSO model checkpoints (EMD loss) |
| `CPD/`      | 100 CPD model checkpoints (EMD loss) |
| `database.json` | Training database (~7,000 DFT-computed α²F) |

To download the models:

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Download everything
huggingface-cli download paprakash/BEE-NET --local-dir BEE-NET-models

# Or download just one variant
huggingface-cli download paprakash/BEE-NET --include "CPD/*" --local-dir BEE-NET-models
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/Train_CSO.ipynb` | Train the CSO model ensemble |
| `notebooks/Train_CPD.ipynb` | Train the CPD model ensemble |
| `notebooks/Pred_CSO.ipynb`  | Run predictions with the CSO ensemble and evaluate |
| `notebooks/Pred_CPD.ipynb`  | Run predictions with the CPD ensemble and evaluate |
| `notebooks/plot_confusion.ipynb` | Generate confusion matrices and precision-recall curves |

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
