# Immunisation Against ITI-attacks

Welcome to the official repository for the paper:

**"E.T.: Ethical Treatment of Language Models Against Harmful Inference-Time Interventions"**

by [Jesus Cevallos](http://www.dista.uninsubria.it/~jesus.cevallos/), [Alessandra Rizzardi](http://www.dista.uninsubria.it/~alessandra.rizzardi/), [Sabrina Sicari](http://www.dicom.uninsubria.it/~sabrina.sicari/), and [Alberto Coen-Porisini](http://www.dicom.uninsubria.it/~alberto.coenporisini/)  
Affiliated with Insubria University.

---

## Overview

This repository contains all the resources you need to reproduce the experiments and results presented in our paper. We provide Jupyter Notebooks for data visualization and scripts to execute our proposed immunisation and stress test procedures on language models.

### Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
  - [Reproduce Plots](#reproduce-plots)
  - [Run Immunisation Procedure](#run-immunisation-procedure)
  - [Run Stress Test](#run-stress-test)
- [Citation](#citation)

---

## Installation and Requirements

Clone the repo:

```bash
git clone https://github.com/DISTA-HCAI/ET
```

Inside the project's main path, clone our fork of the pyREFt library on a directory called "pyreft":


```bash
cd ET

git clone https://github.com/QwertyJacob/my_reft_fork pyreft
```


To run the provided scripts and notebooks, ensure you have the following installed:

- Python 3.8 or higher
- Jupyter Notebook
- Transformers
- Pytorch 2.3


You can install the necessary packages using our script:

```bash
pip install -r requirements.txt
```
(It is recommended to use a virtual env...)
---

## Usage

### Reproduce Plots

To visualize and reproduce the plots from our study, use the Jupyter Notebooks provided.

### Run Immunisation Procedure

To apply the immunisation procedure to a language model, execute the `IMMUNISATION_AND_EVALUATION.sh` script:
(youll need some GPU RAM, notice our code has not been tested on multi-GPU deployments yet... )

This project uses Hydra, we've implemented an overriding mechanism with the keyword param "override" that updates the default base dict with the params in the file with the correspondent name in the config/overrides/ folder.  
Feel free to adjust configs, and create you own override.

1. Make sure the script is executable:
    ```bash
    chmod +x IMMUNISATION_AND_EVALUATION.sh
    ```

2. Run the script:
    ```bash
    ./IMMUNISATION_AND_EVALUATION.sh
    ```

---

## Citation

If you find our work useful in your research, please consider citing our paper:

```
@misc{Cevallos2024,
  title={E.T.: Ethical Treatment of Language Models Against Harmful Inference-Time Interventions},
  author={Cevallos, Jesus and Rizzardi, Alessandra and Sicari, Sabrina and Coen-Porisini, Alberto},
  journal={Tech Report, under review},
  url={https://github.com/DISTA-HCAI/ET}
  year={2025}
  }
```

---

For any questions or issues, please feel free to open an issue on GitHub or contact us via the email 54br1n4.51c4r1{at}un1n5ubr14.1t using this cypher: {s -> 5, 4 -> a, i -> 5 , at -> @})
