# Out-of-Domain Aware Image Classifier

This project implements a convolutional neural network (CNN) for image classification with explicit awareness of out-of-domain (OOD) samples. The goal is to improve model robustness under distribution shift by preventing overconfident predictions on unseen or unrelated data.

Unlike standard classifiers that assume all inputs belong to known classes, this system is trained to distinguish between in-domain and out-of-domain samples using a hybrid loss formulation and uncertainty-aware evaluation.

## Key Features

- Custom CNN architecture implemented in PyTorch for multi-class image classification  
- Hybrid training objective:
  - Cross-entropy loss for labeled in-domain samples  
  - KL-divergence regularization toward a uniform distribution for out-of-domain samples  
  - Curriculum-based weighting to gradually introduce OOD regularization during training  
- Post-training analysis using:
  - Softmax confidence distributions  
  - Prediction entropy  
  - Overconfidence diagnostics under distribution shift  

## Motivation

Deep learning models are often overconfident when exposed to data outside their training distribution, which can lead to unsafe or misleading predictions in real-world systems. This project explores practical techniques to detect and mitigate this failure mode by explicitly modeling uncertainty for unknown inputs.

## Technologies

- Python  
- PyTorch  
- Convolutional Neural Networks (CNNs)  
- NumPy, Matplotlib  


