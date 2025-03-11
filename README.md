Global Ground Metric Learning 
====================
This repository provides the python implementation of the AISTATS2025 paper "Global Ground Metric Learning with Applications to scRNA data" by Damin Kühn nad Michael T. Schaub.

## Abstract
Optimal transport (OT) provides a robust framework for comparing probability distributions. Its effectiveness is significantly influenced by the choice of the underlying ground metric. Traditionally, the ground metric has either been (i) predefined, e.g. as a Euclidean metric, or (ii) learned in a supervised way, by utilizing labeled data to learn a suitable ground metric for enhanced task-specific performance. While predefined metrics often do not account for the inherent structure and varying significance of different features in the data, existing supervised ground metric learning methods often fail to generalize across multiple classes or are limited to distributions with shared supports. To address this issue, this paper introduces a novel approach for learning metrics for arbitrary distributions over a shared metric space. Our method differentiates elements like a global metric, but requires only class labels on a distribution-level for training akin a ground metric. The resulting learned global ground metric enables more accurate OT distances, which can significantly improve clustering and classification tasks. It can create task-specific shared embeddings across elements of different distributions including unseen data.

## Repository Structure
Tutorials on how to use GGML on synthetic or real-world scRNA data are provided as jupyter notebooks in (code).

The classification and clustering experiments are provided as jupyter notebooks in (code/reproduce_experiments).

## Setup
Venv file for convienent use will follow

