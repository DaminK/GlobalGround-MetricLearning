Global Ground Metric Learning 
====================
This repository provides the python implementation to reproduce the results of the AISTATS2025 paper "Global Ground Metric Learning with Applications to scRNA data" by Damin Kühn and Michael T. Schaub.

> [!NOTE]
> To use the GGML method on your own data, refer to the [python package](https://github.com/DaminK/ggml-ot) which you can install via pip:
> ```terminal
> pip install ggml-ot
> ```

## Abstract
Optimal transport (OT) provides a robust framework for comparing probability distributions. Its effectiveness is significantly influenced by the choice of the underlying ground metric. Traditionally, the ground metric has either been (i) predefined, e.g., as a Euclidean metric, or (ii) learned in a supervised way, by utilizing labeled data to learn a suitable ground metric for enhanced task-specific performance.
Yet, predefined metrics typically cannot account for the inherent structure and varying significance of different features in the data, and existing supervised ground metric learning methods often fail to generalize across multiple classes or are limited to distributions with shared supports.   
To address these issues, this paper introduces a novel approach for learning metrics for arbitrary distributions over a shared metric space. 
Our method provides a distance between individual points like a global metric, but requires only class labels on a distribution-level for training. The resulting learned global ground metric enables more accurate OT distances, which can significantly improve embeddings, clustering and classification tasks. We demonstrate the effectiveness and interpretability of our approach using patient-level scRNA-seq data across multiple diseases.

## Repository Structure
Tutorials on how to use GGML on synthetic or real-world scRNA data are provided as Jupyter notebooks in `code/`.

The classification and clustering experiments are provided in `code/reproduce_experiments/`. Use the python script to compute the results and the Jupyter notebooks to visualize the results.

## Setup
To set-up the environment you can use the env file `ggml.yaml`.

