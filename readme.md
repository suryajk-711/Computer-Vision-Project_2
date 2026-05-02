Sources:

- Split same proportion of classes  
  https://medium.com/@aymuosmukherjee/why-do-we-use-stratify-in-train-test-split-e3eb296a5494

- Why normalization  
  https://discuss.pytorch.org/t/should-we-use-our-normalization-for-training-a-pretrained-model/34905

- ImageNet mean and STD  
  https://www.geeksforgeeks.org/python/how-to-normalize-images-in-pytorch/

- To generate only image files  
  https://matplotlib.org/stable/users/explain/figure/backends.html

- Custom Dataset  
  https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html

- Freezing and Unfreezing Layers  
  https://medium.com/we-talk-data/guide-to-freezing-layers-in-pytorch-best-practices-and-practical-examples-8e644e7a9598

- Loading and Saving Checkpoints  
  https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

- Gaussian Blur  
  https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

- Adding Brightness to image  
  https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv

- How to delete a Folder  
  https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder

# Traffic Sign Detection

A classical computer vision pipeline for detecting and classifying traffic signs using HSV color segmentation and SIFT-based template matching.

---

## Overview

This system identifies traffic signs from real-world images by using the fine-tuned EfficientNet B0 model.

Supported classes: `keepRight`, `merge`, `pedestrianCrossing`, `signalAhead`, `speedLimit25`, `speedLimit35`, `stop`, `yield`, `yieldAhead`

---

## Setup

```bash
git clone git@github.com:suryajk-711/Computer-Vision-Project_2.git
cd Computer-Vision-Project_2
pip install -r requirements.txt
```

---

## Usage

**Custom dataset (camera-captured images)**
```bash
python3 src/main.py dataset
```

**Tiny LISA dataset**
```bash
python3 src/main.py db_lisa_tiny
```

**Web frontend**
```bash
python3 src/app.py dataset(to do 3 class classification)
```
OR
```bash
python3 src/app.py db_lisa_tiny(to do 9 class classification)
```
Then open `http://127.0.0.1:5000` and upload an image.

---

**Why EfficientNet-B0?**
--------------------
1. ~5.3M parameters  - small enough for a few-hundred-image dataset
2. Faster inference than ResNet-50 (~82M FLOPs vs ~4B)

---

## Pipeline

<img width="1079" height="606" alt="image" src="https://github.com/user-attachments/assets/cc60a399-3573-47f3-852d-35bcf35d08e4" />

### Step-by-Step

1. **Data Loading** - Images and labels are read from an annotations CSV and stratified into 70 / 15 / 15 train / val / test splits, preserving class proportions across all three sets.

2. **Preprocessing & Augmentation** - All images are resized to 224×224 and normalized using ImageNet mean and standard deviation. For augmented configs, random rotation (±15°), perspective distortion, and color jitter are applied during training to simulate real-world variation in lighting and viewing angle.

3. **Synthetic Image Generation** - For configs 3 and 4, additional synthetic training images are generated per class (e.g: via brightness and blur variations)

4. **Model - EfficientNet-B0** - A pretrained EfficientNet-B0 backbone serves as the feature extractor. A custom classification head (AdaptiveAvgPool => Flatten => Dropout => Linear) is attached on top. The backbone is initially frozen so only the head is trained.

5. **Two-Phase Training** - Phase 1 (epochs 1–10) trains the classification head at a higher learning rate (1e-3) with the backbone frozen. Phase 2 (epoch 11+) unfreezes the last 3 backbone blocks and fine-tunes them at a lower rate (1e-4), letting the model adapt deeper features to traffic sign patterns.

6. **Evaluation** - The best checkpoint per config is evaluated on the held-out test set. Outputs include accuracy, per-class classification report, confusion matrices, misclassified sample grids, and training curves across all four configs.

---

## Results

- **Custom dataset (3 classes):** The model achieved **100% accuracy**, confirming the pipeline is well-designed and the feature extraction is effective for the 3 target classes.

<img width="1000" height="800" alt="image" src="https://github.com/user-attachments/assets/ec399d24-5336-43ac-b509-d0b95608bd3a" />


- **LISA dataset (9 classes):** The model generalizes reasonably well, reaching **87–92% accuracy** despite being trained on a different data distribution - demonstrating the model's transferability across datasets.

<img width="1000" height="800" alt="image" src="https://github.com/user-attachments/assets/5796ed74-de58-4b79-bb11-859957675ba6" />

---

## Limitations

- **No detection component** - The pipeline does not locate signs within a full scene, which restricts its use in real-world deployment where signs appear at arbitrary positions and scales.
