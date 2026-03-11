import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from load_data import load_crownbert_data_from_h5
from model import CrownBERT
from train import train_supervised_model
from test import evaluate_supervised_model


def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # -------------------------
    # 1. Basic settings
    # -------------------------
    set_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = r"UAV_LIDAR24.h5"
    encoder_path = r"pretrain24.pth"

    batch_size = 64
    num_classes = 5
    num_epochs = 50
    learning_rate = 1e-5

    # -------------------------
    # 2. Load data
    # -------------------------
    inputs, attention_mask, position_encoding, labels = load_crownbert_data_from_h5(file_path)

    # -------------------------
    # 3. Normalize input data
    # -------------------------
    mean = inputs.mean(dim=[0, 2, 3], keepdim=True)
    std = inputs.std(dim=[0, 2, 3], keepdim=True)
    inputs = (inputs - mean) / (std + 1e-8)

    # -------------------------
    # 4. Build dataset
    # -------------------------
    dataset = TensorDataset(inputs, position_encoding, attention_mask, labels)

    # -------------------------
    # 5. Split dataset (7:3)
    # -------------------------
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.7)
    test_size = dataset_size - train_size

    generator = torch.Generator().manual_seed(123)
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # -------------------------
    # 6. Initialize model
    # -------------------------
    model = CrownBERT(
        num_transformer_layers=2,
        embed_dim=64,
        num_heads=8,
        feedforward_dim=256,
        in_channels=50,
        height=12,
        width=12,
        num_conv_layers=1,
        reduced_dim=64,
        pretraining_out_channels=1,
        num_classes=num_classes,
        dropout=0.1,
    ).to(device)

    # -------------------------
    # 7. Load pretrained encoder
    # -------------------------
    encoder_state_dict = torch.load(encoder_path, map_location=device)
    model.encoder.load_state_dict(encoder_state_dict)

    # -------------------------
    # 8. Train model
    # -------------------------
    trained_model = train_supervised_model(
        train_loader=train_loader,
        model=model,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        use_attention_mask=True,
        use_position_encoding=True,
    )

    # -------------------------
    # 9. Evaluate model
    # -------------------------
    test_loss, test_accuracy, test_report = evaluate_supervised_model(
        test_loader=test_loader,
        model=trained_model,
        device=device,
        use_attention_mask=True,
        use_position_encoding=True,
    )

    print("\nTest Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("Classification Report:")
    print(test_report)


if __name__ == "__main__":
    main()
