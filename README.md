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

## Requirements

To run the provided scripts and notebooks, ensure you have the following installed:

- Python 3.8 or higher
- Jupyter Notebook
- Transformers
- Pytorch 2.3


You can install the necessary packages using our script:

```bash
source install_requisites.sh
```

---

## Usage

### Reproduce Plots

To visualize and reproduce the plots from our study, use the Jupyter Notebooks provided.

### Run Immunisation Procedure

To apply the immunisation procedure to a language model, execute the `IMMUNISATION_PROCEDURE.sh` script:

1. Make sure the script is executable:
    ```bash
    chmod +x IMMUNISATION_PROCEDURE.sh
    ```

2. Run the script:
    ```bash
    ./IMMUNISATION_PROCEDURE.sh
    ```

### Run Stress Test

To evaluate the immunised model's robustness, run the `STRESS_TEST.sh` script:

1. Ensure the script is executable:
    ```bash
    chmod +x STRESS_TEST.sh
    ```

2. Execute the script:
    ```bash
    ./STRESS_TEST.sh
    ```

---

## Citation

If you find our work useful in your research, please consider citing our paper:

```
@article{Cevallos2024,
  title={E.T.: Ethical Treatment of Language Models Against Harmful Inference-Time Interventions},
  author={Cevallos, Jesus and Rizzardi, Alessandra and Sicari, Sabrina and Coen-Porisini, Alberto},
  journal={ArXiv Preprint},
  year={2024}
  }
```

---

For any questions or issues, please feel free to open an issue on GitHub or contact us via the email 54br1n4.51c4r1{at}un1n5ubr14.1t using this cypher: {s -> 5, 4 -> a, i -> 5 , at -> @})
