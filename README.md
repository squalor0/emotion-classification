# Emotion Classification with CNNs

This project explores facial emotion classification using convolutional neural networks in TensorFlow/Keras. The aim was to build, compare, and evaluate deep learning models that classify face images into emotion categories.

## Overview

I developed a full image-classification workflow, starting with data loading and preprocessing, then moving through baseline modelling, CNN experiments, transfer learning, final evaluation, and visual error analysis.

The notebook includes:

- dataset loading and preprocessing
- class distribution analysis
- majority-class baseline
- CNN model training
- dropout, batch normalisation, augmentation, and regularisation experiments
- MobileNetV2 transfer learning
- validation-based model comparison
- final held-out test evaluation
- confusion matrix and classification report
- misclassified image inspection
- Grad-CAM visualisation

## Main Techniques Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Convolutional Neural Networks
- Transfer Learning
- Grad-CAM

## Model Evaluation

Models were compared using validation performance before selecting a final model. The held-out test set was kept for final evaluation only.

The final analysis includes accuracy, class-level precision/recall/F1, a normalised confusion matrix, and examples of incorrect predictions to better understand where the model struggled.

## Notes

The dataset itself is not included in this repository. The notebook is included to show the modelling process, evaluation approach, and final results.
