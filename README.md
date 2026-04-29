# Rock-Paper-Scissors Image Classification (TensorFlow)

A computer vision project that classifies hand gesture images into **rock**, **paper**, or **scissors** with high validation accuracy using a custom CNN pipeline.

This project was built as a final submission for Dicoding's beginner machine learning track and is designed to demonstrate practical ML engineering skills: data preparation, augmentation strategy, model training, evaluation, and real-time inference.

## Project Snapshot
- **Problem type**: Multi-class image classification (3 classes)
- **Framework**: TensorFlow / Keras
- **Input resolution**: `120x120x3`
- **Dataset size**: **2,188 images**
  - Paper: 712
  - Rock: 726
  - Scissors: 750
- **Split strategy**: per-class split with `train_test_split` (60% train / 40% validation)
  - Train: 1,312 images
  - Validation: 876 images

## Why This Project Is Relevant
- Tackles a classic real-world vision task under constrained data.
- Uses augmentation-heavy training to improve generalization.
- Shows measurable model improvement across repeated runs.
- Includes an interactive prediction flow that simulates a playable "Suwit Jepang" game.

## End-to-End Pipeline
1. **Data acquisition** from Dicoding assets (`rockpaperscissors.zip`).
2. **Per-class splitting** for train and validation sets.
3. **Dataframe-based generators** using `flow_from_dataframe`.
4. **Image augmentation** with random rotation, flips, shear, and zoom.
5. **CNN training** with categorical cross-entropy + Adam optimizer.
6. **Performance tracking** through accuracy/loss curves.
7. **Interactive inference** for user-uploaded images.

## Model Architecture
Primary CNN architecture used in training:
- Conv2D(32, 3x3, ReLU) + MaxPool
- Conv2D(64, 3x3, ReLU) + MaxPool
- Conv2D(128, 3x3, ReLU) + MaxPool
- Conv2D(128, 3x3, ReLU) + MaxPool
- Flatten
- Dense(512, ReLU)
- Dense(3, Softmax)

**Model size**:
- Total parameters: **1,881,283**
- Trainable parameters: **1,881,283**

## Training Configuration
- **Loss**: `categorical_crossentropy`
- **Optimizer**: `Adam`
- **Metric**: `accuracy`
- **Batch size**: 32
- **Epochs**: 20 per run
- **Steps per epoch**: 41
- **Validation steps**: 27

### Augmentation Strategy
- `rescale=1./255`
- `rotation_range=45`
- `horizontal_flip=True`
- `vertical_flip=True`
- `shear_range=0.2`
- `zoom_range=0.2`
- `fill_mode='nearest'`

## Results (Notebook Logs)
| Training Run | Final Train Accuracy | Final Val Accuracy | Best Val Accuracy |
|---|---:|---:|---:|
| Run 1 | 98.63% | 98.96% | 98.96% |
| Run 2 | 98.78% | 98.84% | **99.31%** |

Additional signal from logs:
- Validation performance exceeds 95% early in training and remains stable.
- Final validation loss reaches low values (around `0.033` to `0.056` in final epochs), indicating strong separation between classes.

## Inference Demo
The notebook includes an upload-based prediction flow:
- User uploads a hand-sign image.
- Model predicts one class: `paper`, `rock`, or `scissors`.
- Prediction is used inside a mini "Suwit Jepang" simulation to compare against a random opponent image.

This demonstrates model usability beyond offline metrics.

## Repository Structure
- `Image_classification_paper-rock-scissors.ipynb` - Main notebook (EDA, preprocessing, training, evaluation, inference demo).
- `README.md` - Project documentation.

## How to Run
### Option 1: Google Colab (recommended)
Open and run all cells in:
- `Image_classification_paper-rock-scissors.ipynb`

### Option 2: Local
```bash
pip install tensorflow matplotlib numpy pandas scikit-learn jupyter
jupyter notebook
```
Then run the notebook from top to bottom.

## Skills Demonstrated
- Computer vision data pipeline design
- Robust augmentation for small/medium datasets
- CNN architecture implementation in Keras
- Experiment tracking through training/validation curves
- Practical ML demo integration for end-user interaction

## Author
**Sulhan Fuadi**  
GitHub: https://github.com/sulhanfuadi
