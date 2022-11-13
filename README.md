# Substructure Interaction Graph Network with Node Augmentation for Hybrid Chemical Systems of Heterogeneous Substructures

## Abstract
Complex chemical systems of multiple heterogeneous substructures are common in chemical applications, such as hybrid perovskites and inorganic catalysts. Although graph neural networks (GNNs) have achieved numerous successes in predicting the physical and chemical properties of a single molecule or crystal structure, GNNs for multiple heterogeneous substructures have not yet been studied in chemical science. In this paper, we propose substructure interaction graph network with node augmentation (SIGNNA) that is an integrated architecture of heterogeneous GNNs to predict the physical and chemical properties from the interactions between the heterogeneous substructures of the chemical systems. In addition to the network architecture, we devise a node augmentation method to generate valid subgraphs from given chemical systems for graph-based machine learning, even though the decomposed substructures are physically invalid. SIGNNA outperformed state-of-the-art GNNs in the experimental evaluations and the high-throughput screening on benchmark materials datasets of hybrid organic-inorganic perovskites and inorganic catalysts.

## Run
You can train and evaluate SIGNNA on the benchmark datasets.
The following Python scripts provide the SIGNNA implementation for each benchmark dataset.
- exec_hoip.py: HOIP dataset containing hybrid organic-inorganic perovskites and their band gaps.
- exec_hoip.py: HOIP2d dataset containing hybrid halide perovskites and their band gaps.
- exec_cathub.py: CatHub dataset containing chemical systems of inorganic catalsysts.


## Datasets
We provide the metadata files and the example data of the datasets rather than the full datasets due to the licenses of the benchmark datasets.
You can access the full benchmark datasets throught the following references.
- HOIP dataset: https://www.nature.com/articles/sdata201757
- HOIP2d dataset: https://pubs.acs.org/doi/10.1021/acs.chemmater.0c02290
- CatHub dataset: https://www.nature.com/articles/s41597-019-0081-y


## Employing Chemically Motivated Features
We can customize the initial node features of the virtual nodes in SIGNNA.
To this end, you should implement a function that returns a node-feature vector of ``numpy.array`` for an input ``pymatgen.Structure`` object and an atomic symbol.
Passing the implemented function through ``vn_method`` when calling the ``load_dataset`` or ``load_cathub_dataset`` function to load the dataset.
