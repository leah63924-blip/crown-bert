import copy
import torch
import torch.nn as nn
from sklearn.metrics import classification_report


def train_supervised_model(
    train_loader,
    val_loader,
    model,
    num_epochs: int = 100,
    learning_rate: float = 1e-5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_attention_mask: bool = True,
    use_position_encoding: bool = True,
):
    """
    Supervised training for a model with an already-defined supervised head.

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    model : nn.Module
        Model with the supervised head already configured.
    num_epochs : int, optional
        Number of training epochs.
    learning_rate : float, optional
        Learning rate for optimization.
    device : torch.device, optional
        Device for model training.
    use_attention_mask : bool, optional
        Whether to use attention mask.
    use_position_encoding : bool, optional
        Whether to use positional encoding.

    Returns
    -------
    model : nn.Module
        Best model selected by validation accuracy.
    best_val_accuracy : float
        Best validation accuracy.
    report_dict : dict
        Classification report on the validation set.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

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

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, position_encoding, attention_mask, labels in val_loader:
                inputs = inputs.to(device)
                position_encoding = position_encoding.to(device)
                labels = labels.to(device)
                true_labels = torch.argmax(labels, dim=1)

                if use_attention_mask:
                    attention_mask = attention_mask.to(device)
                else:
                    attention_mask = None

                outputs = model.forward_supervised(
                    x=inputs,
                    position_encoding=position_encoding,
                    key_padding_mask=attention_mask,
                    use_position_encoding=use_position_encoding,
                )

                loss = criterion(outputs, true_labels)
                val_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == true_labels).sum().item()
                total += true_labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total if total > 0 else 0.0

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {avg_val_loss:.4f} "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, position_encoding, attention_mask, labels in val_loader:
            inputs = inputs.to(device)
            position_encoding = position_encoding.to(device)
            labels = labels.to(device)
            true_labels = torch.argmax(labels, dim=1)

            if use_attention_mask:
                attention_mask = attention_mask.to(device)
            else:
                attention_mask = None

            outputs = model.forward_supervised(
                x=inputs,
                position_encoding=position_encoding,
                key_padding_mask=attention_mask,
                use_position_encoding=use_position_encoding,
            )

            predicted = torch.argmax(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

    num_classes = len(set(all_labels))
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=[str(i) for i in range(num_classes)],
        output_dict=True,
    )

    return model, best_val_accuracy, report_dict
