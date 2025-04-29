import os
import torch
import argparse

import torch.nn as nn
import torch.nn.functional as F
import egg.core as core
import matplotlib.pyplot as plt

from .models import Sender, DiscriReceiver, Vision, PretrainNet
from .logger import logger

def pretraining(module: nn.Module, device, train_loader, val_loader, opts, patience=5) -> nn.Module:
    """
    Pretrains the given vision module with a classification task, using early stopping to prevent overfitting.

    Args:
        module (nn.Module): The vision module to pretrain.
        device (torch.device): Device to run training on (e.g., 'cuda' or 'cpu').
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        opts (argparse.Namespace): Command-line arguments.
        patience (int): Number of epochs to wait for improvement before early stopping.


    Returns:
        nn.Module: The pretrained vision module with best weights.
    """
    epochs = opts.pretrain_epochs

    if epochs == 0:
        return module

    logger.info(f"Pretraining Vision Module for up to {epochs} epochs with early stopping")
    class_prediction = PretrainNet(module)

    optimizer = core.build_optimizer(class_prediction.parameters())
    class_prediction = class_prediction.to(device)

    train_loss_values = []
    train_accuracy_values = []
    val_loss_values = []
    val_accuracy_values = []

    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    location = opts.vision_to_file

    def evaluate(model, data_loader):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        n_batches = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.cross_entropy(output, target.long())

                total_loss += loss.item()
                n_batches += 1

                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)

        return total_loss / n_batches, 100 * correct / total

    for epoch in range(epochs):
        class_prediction.train()
        mean_loss, n_batches, correct, total = 0, 0, 0, 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = class_prediction(data)
            loss = F.cross_entropy(output, target.long())
            loss.backward()
            optimizer.step()

            mean_loss += loss.item()
            n_batches += 1

            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

        train_loss = mean_loss / n_batches
        train_accuracy = 100 * correct / total

        val_loss, val_accuracy = evaluate(class_prediction, val_loader)

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
        )

        train_loss_values.append(train_loss)
        train_accuracy_values.append(train_accuracy)
        val_loss_values.append(val_loss)
        val_accuracy_values.append(val_accuracy)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = class_prediction.vision_module.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            if val_accuracy < 0.9:
                logger.info("Early stopping triggered before model could learn, retrying")
                return pretraining(module, device, train_loader, val_loader, epochs, patience)
            break

    fig = plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, marker='o', label='Train')
    plt.plot(range(1, len(val_loss_values) + 1), val_loss_values, marker='o', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(train_accuracy_values) + 1), train_accuracy_values, marker='o', label='Train')
    plt.plot(range(1, len(val_accuracy_values) + 1), val_accuracy_values, marker='o', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.subplot(1, 3, 3)
    train_val_loss_diff = [t - v for t, v in zip(train_loss_values, val_loss_values)]
    plt.plot(range(1, len(train_val_loss_diff) + 1), train_val_loss_diff, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Train-Val Loss Difference')
    plt.title('Overfitting Indicator')

    plt.tight_layout()

    directory=f'./src/data/logs/{opts.run_uuid}/plots'
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'{directory}/pretraining_plot_{opts.run_uuid}.png')
    plt.close(fig)

    # Load best weights
    module.load_state_dict(best_weights)

    if location != '':
        logger.info(f"Saving model to {location}")
        torch.save(module.state_dict(), location)

    return module
