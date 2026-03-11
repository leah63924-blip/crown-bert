import h5py
import numpy as np
import torch


def load_crownbert_data_from_h5(file_path: str):
    """
    Load Crown-BERT input data from an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the input HDF5 file.

    Returns
    -------
    inputs : torch.Tensor
        Hyperspectral input tensor with shape (N, B, H, W).
    attention_mask : torch.Tensor
        Crown attention mask with shape (N, H, W).
    position_encoding : torch.Tensor
        Crown positional encoding with shape (N, H, W).
    labels : torch.Tensor
        Label tensor with shape (N, C) or the corresponding label format stored in the HDF5 file.
    """
    with h5py.File(file_path, "r") as f:
        inputs = torch.tensor(np.array(f["inputs"]), dtype=torch.float32)
        attention_mask = torch.tensor(np.array(f["attention_mask"]), dtype=torch.float32)
        position_encoding = torch.tensor(np.array(f["position_encoding"]), dtype=torch.float32)
        labels = torch.tensor(np.array(f["labels"]), dtype=torch.float32)

    return inputs, attention_mask, position_encoding, labels
