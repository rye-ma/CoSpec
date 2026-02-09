# CoSpec: Unified Cross-City Road Representation Learning
This repository contains the official implementation of **Seeking Commonality, Preserving Specificity: A Spectral-Aware Hierarchical Framework for Cross-City Road Representation Learning (CoSpec)** submitted to ICML 2026.

## 1. Overview
CoSpec aims to learn transferable road network representations across different cities (e.g., Beijing, Chengdu, Xi'an) with a framework that disentangles road networks into shareable low-frequency commonalities and city-specific high-frequency specificities. The framework consists of a Pre-training phase that aligns macro-semantic patterns across cities using prototypes, followed by Downstream Tasks to evaluate the effectiveness of the learned representations.

## 2. Environment Setup
The code is implemented in Python 3.10 and PyTorch 2.8.0. We recommend using Conda to manage the environment.

## 3. Data Preparation
### 3.1 Data Source
The raw road network and trajectory data used in this project are based on the HRNR dataset. Please download the data from the following repository:
[Download Link](https://gitee.com/solaris_wn/HRNR)

### 3.2 Directory Structure
To ensure the configuration files load the data correctly, please organize the downloaded data into a root-level data/ directory.
The expected directory structure is:
```
CoSpec/
├── data/                  <-- Create this folder in the root
│   ├── bj/                <-- Place Beijing data files here
│   │   ├── train_route_set
│   │   ├── test_route_set
│   │   └── ...
│   ├── cd/                <-- Place Chengdu data files here
│   └── xa/                <-- Place Xi'an data files here
├── beijing/               <-- City-specific code (Source Code)
├── chengdu/
├── xian/
├── pretrain/
└── README.md
```

Note: The configuration files (e.g., beijing/configs/route_plan.yml) are set to look for data at ./data/bj/... by default.

## 4. Usage
### 4.1 Pre-training
The pre-training phase learns the unified spectral representations using data from all cities.
```
python pretrain/pretrain.py --config pretrain/configs/pretrain.yml
```

### 4.2 Downstream Tasks
After pre-training, you can fine-tune or evaluate the model on specific downstream tasks for each city.
```
python beijing/downstream_task/train_route_plan.py --config beijing/configs/route_plan.yml
```

## 5. Anonymity Statement
- This repository is anonymized for ICML 2026 Double-Blind Review.
- Author names, affiliations, and acknowledgments have been removed.
- All data paths are relative; no absolute paths to local servers are included.
