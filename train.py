import torch
import torch.nn as nn
def train_supervised_model(
    train_loader,
    model,
    num_epochs: int = 100,
    learning_rate: float = 1e-5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_attention_mask: bool = True,
    use_position_encoding: bool = True,
):
    """
    Train a supervised Crown-BERT model.

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader.
    model : nn.Module
        Model with the supervised head already configured.
    num_epochs : int, optional
        Number of training epochs.
    learning_rate : float, optional
        Learning rate for optimization.
    device : torch.device, optional
        Device for training.
    use_attention_mask : bool, optional
        Whether to use attention mask.
    use_position_encoding : bool, optional
        Whether to use positional encoding.

    Returns
    -------
    model : nn.Module
        Trained model.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, position_encoding, attention_mask, labels in train_loader:
            inputs = inputs.to(device)
            position_encoding = position_encoding.to(device)
            labels = labels.to(device)
            labels = torch.argmax(labels, dim=1)

            if use_attention_mask:
                attention_mask = attention_mask.to(device)
            else:
                attention_mask = None

            optimizer.zero_grad()

            outputs = model.forward_supervised(
                x=inputs,
                position_encoding=position_encoding,
                key_padding_mask=attention_mask,
                use_position_encoding=use_position_encoding,
            )

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f}"
        )

    return model
