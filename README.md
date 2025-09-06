# Landmark Classification with CNNs

## 📌 Overview

This project is part of the **Udacity Deep Learning Nano Degree** program.
The goal is to build a **landmark image classifier** that predicts the most likely location of a photo by recognizing distinctive landmarks, even when GPS metadata is unavailable.

The project demonstrates the **end-to-end deep learning workflow**:

* Preprocessing real-world image data
* Building and training Convolutional Neural Networks (CNNs) from scratch
* Applying transfer learning with pretrained models
* Comparing model performance
* Deploying the best-performing model in an interactive app that accepts user-uploaded images

---

## 🛠️ Tech Stack

* Python 3.7+
* PyTorch – deep learning framework
* Torchvision – pretrained models & data transforms
* NumPy, Pandas, Matplotlib – data wrangling & visualization
* Jupyter Notebook – interactive development

---

## 📊 Dataset

* Subset of the **Google Landmarks Dataset v2**
* Includes **50 landmark categories** from across the world
* Dataset provided by Udacity (not included in this repo due to size)
* [Dataset info](https://github.com/cvdfoundation/google-landmark)

---

## 🚀 Project Workflow

1. **Data Preprocessing**

   * Image resizing, normalization, and augmentation
   * Splitting into train/validation/test sets

2. **Model Development**

   * CNN from scratch
   * Transfer learning with pretrained architectures (ResNet)
   * Hyperparameter tuning (learning rate, batch size, optimizer, etc.)

3. **Evaluation & Comparison**

   * Loss Function: Cross Entropy Loss
   * Validation Metric: Top-1 Accuracy
   * Training vs. validation loss curves for each experiement

4. **Deployment**

   * Notebook app that takes an input image
   * Returns **top-5 predicted landmarks** with probabilities

---

## 📈 Results


* **From Scratch CNN**: Established baseline accuracy
* **Transfer Learning (ResNet)**: Achieved significantly higher accuracy and generalization
* Final app successfully predicts top-5 landmarks for unseen images

<img width="2214" height="866" alt="لقطة الشاشة 2025-09-03 222046" src="https://github.com/user-attachments/assets/2cd0ef20-03fb-49c5-8e78-07680cf6a15c" />

---

## 📂 Repository Structure

```
.
├── README.md
├── requirements.txt
├── app.ipynb                     # Notebook app: load image → predict top-5 landmarks
├── cnn_from_scratch.ipynb        # Baseline CNN built/trained from scratch
├── transfer_learning.ipynb       # Experiments with pretrained backbone (ResNet)
├── workspace_utils.py            # Utility helpers
├── mean_and_std.pt               # Cached dataset normalization stats (tensor file)
├── proof.png                     # Sample output / evidence image for README
├── static_images/                # Images used in README/docs (figures, screenshots)
├── src/
│   ├── __init__.py
│   ├── data.py                   # Datasets, transforms, loaders, augmentation code
│   ├── helpers.py                # Generic utilities 
│   ├── model.py                  # CNN definitions (scratch architectures)
│   ├── optimization.py           # Optimizers, schedulers, training hyperparams
│   ├── predictor.py              # Inference utilities (top-5 predictions, label mapping)
│   ├── train.py                  # Training loop(s): fit/validate, checkpoints, logging
│   └── transfer.py               # Transfer-learning model / head replacements

```

---

## 🎓 Acknowledgements

* Project completed as part of the **Udacity Deep Learning Nano Degree**
* Dataset: [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark)
