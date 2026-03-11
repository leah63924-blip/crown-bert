# Crown-BERT

Code and reproducibility resources for Crown-BERT for individual tree species classification using UAV hyperspectral and LiDAR data.

Crown-BERT is a deep learning framework for individual tree species classification using UAV hyperspectral and LiDAR data.

It is designed for crown-level classification by jointly leveraging spectral and structural information. The framework integrates Dynamic Crown Masking (DCM), Crown Positional Encoding (CPE), and Crown Masked Pixel Modeling (CMPM) within a BERT-style architecture. This repository provides the main implementation of Crown-BERT together with the associated resources for training, evaluation, and reproducibility.
<img width="976" height="817" alt="image" src="https://github.com/user-attachments/assets/b8f0b7b5-aab4-4cf1-80a5-f57bc1c2894e" />
**Figure.** Overall workflow of Crown-BERT, including crown-level input construction, morphology-aware representation with DCM and CPE, self-supervised pre-training with CMPM, and supervised fine-tuning for individual tree species classification.
<img width="1005" height="461" alt="image" src="https://github.com/user-attachments/assets/f718f147-b558-4a20-865f-a9850930c969" />
**Figure.** Detailed architecture of Crown-BERT. The framework integrates Dynamic Crown Masking (DCM), Crown Positional Encoding (CPE), and Crown Masked Pixel Modeling (CMPM) within a hybrid convolution–Transformer backbone for crown-scale individual tree species classification.
## Data Organization

The input dataset is organized in **HDF5 (`.h5`)** format. Each HDF5 file contains four keys corresponding to the crown-level inputs and labels used by Crown-BERT:

- `inputs`: hyperspectral patch data, with shape `(N, B, H, W)` and data type `float32`
- `attention_mask`: crown-region mask, with shape `(N, H, W)` and data type `float32`
- `position_encoding`: crown positional encoding map, with shape `(N, H, W)` and data type `float32`
- `labels`: one-hot encoded class labels, with shape `(N, C)` and data type `float32`

Here, `N` denotes the number of crown samples, `B` denotes the spectral dimension, `H × W` denotes the spatial patch size, and `C` denotes the number of classes. The `inputs` key stores the hyperspectral crown patches, `attention_mask` indicates valid crown regions, `position_encoding` provides the crown positional prior, and `labels` stores the class annotations for supervised training and evaluation.
## Code Structure

The repository mainly consists of modules for data loading, model definition, training, and testing.

- `load_data.py`: functions for reading HDF5 files and preparing the crown-level inputs used by Crown-BERT  
- `model.py`: implementation of the Crown-BERT network architecture  
- `train.py`: training script for model optimization  
- `test.py`: evaluation script for model testing and performance assessment  
- `main.py`: main script for organizing the overall workflow, including data loading, model construction, training, and evaluation
## Usage

The environment configuration, dependency settings, and key hyperparameters used in this project are provided directly in the code. Please refer to `train.py`, `test.py`, and the related implementation files for details.
