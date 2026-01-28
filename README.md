# Deforestation Detection via Metric Learning

A deep learning framework for detecting deforestation in satellite imagery using metric learning and embedding-based classification. This project leverages PyTorch, PyTorch Lightning, and PyTorch Metric Learning to train models that learn discriminative embeddings for distinguishing between forest and recently deforested regions.

## Overview

This repository implements a multi-stage pipeline for deforestation detection:
Our goal is to use deep metric learning to create robust embeddings for superpixel segments extracted from satellite images, which can then be classified into forested and deforested (will be expanded to other deforestation types) areas.
1. **Dataset Preparation**: Extract and preprocess superpixel segments from satellite imagery
2. **Deep Metric Learning**: Train embedding models using triplet loss to learn discriminative features
3. **Embedding Extraction**: Generate embeddings from trained models (ResNet or Haralick features)
4. **Classification**: Train classifiers (SVM or neural networks) on the learned embeddings

## Installation

```bash
# Clone the repository
git clone https://github.com/ebouhid/deforestation-metric-learning.git
cd deforestation-metric-learning

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

- PyTorch >= 2.5.1
- PyTorch Lightning >= 2.5.0
- PyTorch Metric Learning >= 2.8.1
- timm >= 1.0.13 (PyTorch Image Models)
- scikit-learn >= 1.6.1
- UMAP-learn >= 0.5.7
- albumentations >= 2.0.0
- mahotas >= 1.4.18 (for Haralick features)

## Workflow

### 1. Dataset Preparation

Prepare the dataset by extracting superpixel segments from Landsat-8 satellite imagery:

```bash
python build_dataset.py
```

This script:
- Loads satellite scenes and ground truth masks
- Extracts superpixel segments using SLIC segmentation
- Filters segments by Homogeneity Ratio (HoR ≥ 0.7)
- Applies data augmentation (rotation)
- Saves segments as 64×64 PNG images organized by region

**Data Split** (by region):
- **Training**: x01, x02, x06, x07, x08, x10
- **Validation**: x09
- **Test**: x03, x04

### 2. Deep Metric Learning Training

Train embedding models using triplet margin loss:

```bash
python metric_learning.py
```

**Supported Backbones**:
- ResNet-18
- ResNet-50
- ResNet-101

**Training Configuration**:
- Loss: Triplet Margin Loss (margin=0.2)
- Miner: TripletMarginMiner
- Sampler: MPerClassSampler (balanced batches)
- Optimizer: Adam (lr=0.0001)
- Epochs: 200

The training process generates UMAP visualizations at each epoch to monitor embedding quality.

### 3. Embedding Extraction

Extract embeddings from trained models or pretrained backbones:

```bash
# Using fine-tuned DML model
python get_embeddings.py \
    --model_name resnet18 \
    --input_dir segment_embeddings_classification_dataset_norsz/ \
    --output_dir embeddings_r18dml/ \
    --fine_tune_ckpt dml_results/resnet18/best_model_state_dict_1.pth

# Using pretrained model (no fine-tuning)
python get_embeddings.py \
    --model_name resnet50 \
    --input_dir segment_embeddings_classification_dataset_norsz/ \
    --output_dir embeddings_r50/

# Using Haralick texture features
python get_embeddings.py \
    --model_name haralick \
    --input_dir segment_embeddings_classification_dataset_norsz/ \
    --output_dir embeddings_har/
```

Or run the batch script:
```bash
bash get_embeddings.sh
```

### 4. Classification

#### Option A: SVM Classification

Run SVM grid search to find optimal hyperparameters:

```bash
python svm_gridsearch_eval.py
```

**Grid Search Parameters**:
- C: [0.05, 0.1, 1, 10, 100, 1000]
- Kernel: ['rbf', 'linear', 'poly']
- Gamma: [0.1, 1, 'auto']

Results are saved to `svm_gridsearch_results.csv`.

#### Option B: Neural Network Classification

Train end-to-end classification networks:

```bash
python train_classification_networks.py
```

This trains ResNet-18 and ResNet-50 models with:
- Binary Cross-Entropy loss
- Adam optimizer (lr=0.001)
- Model checkpointing based on validation F1 score

### 5. Visualization

Generate training metric plots:
```bash
python get_plots.py
```

Create UMAP embedding videos:
```bash
python umap_to_video.py -i dml_results_654/resnet18/umap_plots -o umap_videos/r18_umap.mp4 -f 30
```

## Classes

The dataset contains two main classes:
- **Forest** (label 0): Preserved forest regions
- **Recent Deforestation** (label 1): Areas with recent deforestation activity

## Metrics

The project evaluates models using:
- F1 Score
- Precision
- Recall
- Balanced Accuracy
- True Negative Rate (TNR)

## Results

Results are stored in:
- `svm_gridsearch_results.csv` - SVM hyperparameter search results
- `network_clf_best_results.csv` - Neural network classification results
- `test_results.csv` - Final test set evaluation
- `network_training_results/` - Training curves and metrics

## Data Organization

### Input Data Format
- Satellite imagery: NumPy arrays with multiple spectral bands (Landsat-8, Sentinel-2 coming soon)
- Ground truth: Binary masks (0=not analyzed, 1=recent deforestation, 2=forest)
- Superpixels: SLIC segmentation maps

### Output Format
- Segments: 64×64 RGB PNG images
- Embeddings: NumPy arrays (.npy files)
- Models: PyTorch state dictionaries and joblib-serialized SVMs


## Acknowledgments

- ForestEyes project for satellite imagery data💚
