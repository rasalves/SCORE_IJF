# SCORE: A Convolutional-based Approach for Football Event Forecasting

## Abstract

Football (also known as Soccer or Association Football), the most popular sport in the world, is a blend of skill and luck, making it highly unpredictable. To address this unpredictability, there has been a surge in popularity in employing machine learning techniques for forecasting football-related features over the past decade. This trend aligns with the growing professionalism in football analytics. Despite this progress, the existing body of work remains in its early stages, lacking the depth required to capture the intricate nuances of the sport. In this study, we introduce a convolution-based approach designed to predict the occurrence of the next event in a football match, such as a goal or a corner kick, relying solely on easy-to-access past events for predictions. Our methodology adopts an online approach, meaning predictions can be computed during a live match. To validate our approach, we conduct a comprehensive evaluation against five baseline models, utilizing data from various elite European football leagues. Additionally, an ablation study is performed to understand the underlying mechanisms of our method. Finally, we present practical applications and interpretable aspects of our proposed approach.


## Date of Assembly

This reproducibility package was completed on **November 29, 2024**. The routines have been adapted to allow experiments to be reproduced in any computing environment, ensuring flexibility and applicability to a variety of setups.

## Author

**Rodrigo Alves**\
For further information, please contact: [rodrigo.alves@fit.cvut.cz](mailto\:rodrigo.alves@fit.cvut.cz)

## Structure of the Repository

The repository is organized to facilitate reproducibility of the experiments. The provided Python script `source/001_createDirectories.py` ensures that the required directory structure is created. Below is the main structure:

- **source/**: Contains folders, Python scripts, R scripts, and Jupyter notebooks to reproduce the experiments.
  - **source/DATA**: This folder is intended to hold the dataset files. The dataset used in this paper is publicly available on Kaggle ([Football Events Dataset](https://kaggle.com/datasets/secareanualin/football-events)). To reproduce the experiments, download and extract the files into this directory. Alternatively, use the provided script `source/002_downloadData.py`.
  - **source/FIGURES**: Stores PDF files of the generated figures.
  - **source/RES**: Contains intermediate results and final outputs. Due to the considerable time required for computation and the large size of intermediate files, key results are saved here to allow direct reproduction of figures and tables.
  - **source/TRANSFER**: Holds files related to transfer learning.
  - **source/YS**: Stores label files.

## Computing Environment

### Languages and Packages

#### R

The following R packages are required:

| Package      |
| ------------ |
| reticulate   |
| RColorBrewer |
| readr        |

#### Python

The following Python packages and versions are required:

| Package      | Version |
| ------------ | ------- |
| kagglehub    | 0.3.4   |
| keras        | 3.3.3   |
| numpy        | 1.26.4  |
| pandas       | 2.2.2   |
| scikit-learn | 1.5.0   |
| scipy        | 1.13.1  |
| tensorflow   | 2.16.1  |
| tqdm         | 4.66.4  |

### Experimental Environment

Most experiments were conducted on a high-performance cluster with the following specifications:
- **CPU**: 128 cores (Intel Xeon Gold 6254 @ 3.10GHz)
- **RAM**: 512 GB
- **GPU**: DGX Station A100 for deep learning baselines

The cluster configuration:
- Architecture: x86_64
- NUMA nodes: 8
- Caches: L1d (32K), L1i (32K), L2 (1024K), L3 (25344K)

A virtual Python environment was created, and packages were installed via `pip`. Running experiments for hyperparameter selection took approximately 48 hours per league. A single run with fixed hyperparameters takes only a few minutes. Complex deep learning baselines (e.g., LSTM) were run using a DGX Station A100.

## Data Preprocessing

The dataset is publicly available on Kaggle ([Football Events Dataset](https://kaggle.com/datasets/secareanualin/football-events)). The preprocessing steps can be replicated using the script `source/003_cleanData.py`.

## Figures and Tables

### Figures

All result-driven figures were reproduced. The output files are stored in the `source/FIGURES` folder. Most figures require preprocessing steps. For example:

- To generate **Figure 2**, run `source/figure2.py` first, then use the corresponding Jupyter notebook to create the final visualization.
- All figures were created using R.

### Tables

Tables have corresponding Python scripts for reproduction. For example:

- **Table 3**: Run `source/table3.py` to reproduce the results.

## Special Steps

Key results from the experiments have been saved to simplify viewing. For a full reproduction of all experiments, the following files should be executed in order:

1. `001_createDirectories.py`
2. `002_downloadData.py`
3. `003_cleanData.py`
4. `004_run_forecast.py`
5. `005_run_transfer.py`
6. `006_run_ablation.py`


Afterward, all figures and tables can be generated directly from the provided scripts and notebooks.

