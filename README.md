# Substructure Interaction Graph Network with Node Augmentation for Hybrid Chemical Systems of Heterogeneous Substructures

## Summary
Complex chemical systems of multiple heterogeneous substructures are common in chemical applications, such as hybrid perovskites and inorganic catalysts. Although graph neural networks (GNNs) have achieved numerous successes in predicting the physical and chemical properties of a single molecule or crystal structure, GNNs for multiple heterogeneous substructures have not yet been studied in chemical science. In this paper, we propose substructure interaction graph network with node augmentation (SIGNNA) that is an integrated architecture of heterogeneous GNNs to predict the physical and chemical properties from the interactions between the heterogeneous substructures of the chemical systems. In addition to the network architecture, we devise a node augmentation method to generate valid subgraphs from given chemical systems for graph-based machine learning, even though the decomposed substructures are physically invalid. SIGNNA outperformed state-of-the-art GNNs in the experimental evaluations and the high-throughput screening on benchmark materials datasets of hybrid organic-inorganic perovskites and inorganic catalysts.

## Run
You can train and evaluate SIGNNA on the benchmark datasets.
The following Python scripts provide the SIGNNA implementation for each benchmark dataset.
- exec_hoip.py: HOIP dataset.
- exec_hoip.py: HOIP2d dataset.
- exec_cathub.py: CatHub dataset.
