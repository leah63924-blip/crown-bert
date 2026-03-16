import torch
import torch.nn as nn
from sklearn.metrics import classification_report


def evaluate_supervised_model(
    test_loader,
    model,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_attention_mask: bool = True,
    use_position_encoding: bool = True,
):
    """
    Evaluate a supervised Crown-BERT model on the test set.

    Parameters
    ----------
    test_loader : DataLoader
        Test data loader.
    model : nn.Module
        Model with the supervised head already configured.
    device : torch.device, optional
        Device for evaluation.
    use_attention_mask : bool, optional
        Whether to use attention mask.
    use_position_encoding : bool, optional
        Whether to use positional encoding.

    Returns
    -------
    test_loss : float
        Average test loss.
    test_accuracy : float
        Test accuracy.
    report_dict : dict
        Classification report on the test set.
    """
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, position_encoding, attention_mask, labels in test_loader:
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
            test_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == true_labels).sum().item()
            total += true_labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct / total if total > 0 else 0.0

    num_classes = len(set(all_labels))
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=[str(i) for i in range(num_classes)],
        output_dict=True,
    )

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    return avg_test_loss, test_accuracy, report_dict
