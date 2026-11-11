# CoMet-Industrial-Anomaly-Detection-Thesis
Confident Meta-Learning for Unsupervised Anomaly Detection on Real-World Industrial Data
Project Overview:
This repository contains the implementation and experimental results of my Master's thesis:
"Confident Meta-Learning for Unsupervised Anomaly Detection on Real-World Industrial Data."
The project investigates the use of confidence-aware meta-learning for unsupervised industrial anomaly detection.
The proposed CoMet approach explores how data confidence and model confidence can contribute to reliable anomaly detection and localization in real-world industrial inspection scenarios.
The experiments were conducted using the MVTec AD 2 dataset, which provides challenging industrial inspection images under different conditions.
 Research Objectives
The main objectives of this research are:
Investigate confidence-aware meta-learning for unsupervised anomaly detection.
Evaluate the contribution of data confidence and model confidence.
Detect anomalous patterns in industrial images.
Localize anomalies through pixel-level anomaly maps.
Analyse model performance under different levels of training-data noise.
Compare the complete CoMet approach with different ablation configurations.

Methodology
The project implements CoMet (Confident Meta-Learning), exploring confidence mechanisms within an unsupervised anomaly detection framework.
The implementation uses WideResNet-50 as the feature extraction backbone.
The main research components include:
Feature extraction using WideResNet-50.
Confidence-aware learning mechanisms.
Data confidence evaluation.
Model confidence evaluation.
Dataset
Experiments were performed using the MVTec AD 2 dataset.
The dataset includes eight industrial categories used in this research:
Can,Fabric,Rice,Vial,Fruit Jelly,Wallplugs,Walnuts,Sheet Metal

Experimental Configurations

Four configurations were evaluated to investigate the contribution of the confidence mechanisms.

Configuration

Description

Full CoMet

Complete model with data and model confidence

No Data Confidence

Model without data confidence

No Model Confidence

Model without model confidence

No Data + No Model Confidence

Model without both confidence mechanisms

These ablation experiments help analyse the individual and combined contributions of the confidence components.

📊 Evaluation Metrics

The proposed approach was evaluated using three main metrics:

1. Image AUROC

Measures the model's ability to distinguish normal images from anomalous images.

2. Pixel AUROC

Measures the accuracy of pixel-level anomaly localization.

3. PRO (Per-Region Overlap)

Evaluates the quality of anomaly localization by measuring overlap between predicted anomaly regions and ground-truth regions.

Noise Robustness Experiments

An additional experiment investigated the impact of noisy training data on anomaly detection performance.

Different training-data noise ratios were evaluated:
0%, 2% ,4%, 6%,8%,10%
The experiments analyse how increasing training-data noise affects anomaly detection performance and model robustness.
