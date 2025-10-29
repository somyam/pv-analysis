"""
Training script for drug-reaction association prediction models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
import time

from model_with_features import get_model


class DrugReactionDataset(Dataset):
    """PyTorch Dataset for drug-reaction pairs."""

    def __init__(
        self,
        drug_indices: np.ndarray,
        reaction_indices: np.ndarray,
        labels: np.ndarray,
    ):
        self.drug_indices = torch.from_numpy(drug_indices).long()
        self.reaction_indices = torch.from_numpy(reaction_indices).long()
        self.labels = torch.from_numpy(labels).float().unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.drug_indices[idx],
            self.reaction_indices[idx],
            self.labels[idx],
        )


def load_data(data_dir: str) -> Tuple[Dict, Dict]:
    """Load processed data and metadata."""
    data_dir = Path(data_dir)

    # Load splits
    datasets = {}
    for split in ["train", "val", "test"]:
        data = np.load(data_dir / f"{split}.npz")
        datasets[split] = DrugReactionDataset(
            data["drug_indices"], data["reaction_indices"], data["labels"]
        )

    # Load metadata
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    return datasets, metadata


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for drug_idx, reaction_idx, labels in dataloader:
        drug_idx = drug_idx.to(device)
        reaction_idx = reaction_idx.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(drug_idx, reaction_idx)
        loss = criterion(predictions, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0
    n_batches = 0

    for drug_idx, reaction_idx, labels in dataloader:
        drug_idx = drug_idx.to(device)
        reaction_idx = reaction_idx.to(device)
        labels = labels.to(device)

        predictions = model(drug_idx, reaction_idx)
        loss = criterion(predictions, labels)

        all_predictions.append(predictions.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_loss += loss.item()
        n_batches += 1

    # Concatenate all predictions and labels
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    # Compute metrics
    avg_loss = total_loss / n_batches
    roc_auc = roc_auc_score(all_labels, all_predictions)
    avg_precision = average_precision_score(all_labels, all_predictions)

    # Accuracy with threshold 0.5
    binary_preds = (all_predictions > 0.5).astype(int)
    accuracy = (binary_preds == all_labels).mean()

    return {
        "loss": avg_loss,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def plot_metrics(train_history: Dict, val_history: Dict, save_dir: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = ["loss", "roc_auc", "avg_precision", "accuracy"]
    titles = ["Loss", "ROC AUC", "Average Precision", "Accuracy"]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        epochs = range(1, len(train_history[metric]) + 1)
        ax.plot(epochs, train_history[metric], label="Train", marker="o")
        ax.plot(epochs, val_history[metric], label="Val", marker="s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{title} over epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close()


def plot_roc_curve(labels: np.ndarray, predictions: np.ndarray, save_dir: Path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = roc_auc_score(labels, predictions)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Drug-Reaction Association Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "roc_curve.png", dpi=150)
    plt.close()


def plot_pr_curve(labels: np.ndarray, predictions: np.ndarray, save_dir: Path):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(labels, predictions)
    avg_precision = average_precision_score(labels, predictions)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall, precision, label=f"PR curve (AP = {avg_precision:.3f})", linewidth=2
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Drug-Reaction Association Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "pr_curve.png", dpi=150)
    plt.close()


def train_model(
    model_type: str = "embedding_mlp",
    data_dir: str = "data/processed",
    output_dir: str = "outputs/simple_dl",
    # Model hyperparameters
    embedding_dim: int = 64,
    hidden_dims: Optional[list] = None,
    dropout: float = 0.3,
    # Training hyperparameters
    batch_size: int = 512,
    learning_rate: float = 0.001,
    n_epochs: int = 50,
    weight_decay: float = 1e-5,
    patience: int = 10,
    # Device
    device: str = "cpu",
):
    """
    Main training function.

    Args:
        model_type: Type of model to train
        data_dir: Directory with processed data
        output_dir: Where to save model and results
        embedding_dim: Dimension of embeddings
        hidden_dims: Hidden layer dimensions for MLP
        dropout: Dropout probability
        batch_size: Batch size for training
        learning_rate: Learning rate
        n_epochs: Maximum number of epochs
        weight_decay: L2 regularization weight
        patience: Early stopping patience
        device: "cpu" or "cuda"
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load data
    print("Loading data...")
    datasets, metadata = load_data(data_dir)
    n_drugs = metadata["n_drugs"]
    n_reactions = metadata["n_reactions"]

    print(f"Data loaded:")
    print(f"  Drugs: {n_drugs}")
    print(f"  Reactions: {n_reactions}")
    print(f"  Train samples: {len(datasets['train'])}")
    print(f"  Val samples: {len(datasets['val'])}")
    print(f"  Test samples: {len(datasets['test'])}\n")

    # Create dataloaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    print(f"Creating {model_type} model...")
    model = get_model(
        model_type=model_type,
        n_drugs=n_drugs,
        n_reactions=n_reactions,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    # Training history
    train_history = {"loss": [], "roc_auc": [], "avg_precision": [], "accuracy": []}
    val_history = {"loss": [], "roc_auc": [], "avg_precision": [], "accuracy": []}

    best_val_auc = 0
    patience_counter = 0
    best_epoch = 0

    print("Starting training...\n")
    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate on train and val
        train_metrics = evaluate(model, train_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Update history
        for metric in ["loss", "roc_auc", "avg_precision", "accuracy"]:
            train_history[metric].append(train_metrics[metric])
            val_history[metric].append(val_metrics[metric])

        # Learning rate scheduling
        scheduler.step(val_metrics["roc_auc"])

        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{n_epochs} ({epoch_time:.1f}s)")
        print(
            f"  Train: loss={train_metrics['loss']:.4f}, "
            f"AUC={train_metrics['roc_auc']:.4f}, "
            f"AP={train_metrics['avg_precision']:.4f}, "
            f"acc={train_metrics['accuracy']:.4f}"
        )
        print(
            f"  Val:   loss={val_metrics['loss']:.4f}, "
            f"AUC={val_metrics['roc_auc']:.4f}, "
            f"AP={val_metrics['avg_precision']:.4f}, "
            f"acc={val_metrics['accuracy']:.4f}"
        )

        # Save best model
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                output_dir / "best_model.pt",
            )
            print(f"  ✓ New best model saved (val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1

        print()

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    # Load best model for final evaluation
    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  ROC AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  Average Precision: {test_metrics['avg_precision']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")

    # Plot results
    print("\nGenerating plots...")
    plot_metrics(train_history, val_history, output_dir)
    plot_roc_curve(test_metrics["labels"], test_metrics["predictions"], output_dir)
    plot_pr_curve(test_metrics["labels"], test_metrics["predictions"], output_dir)

    # Save results
    results = {
        "model_type": model_type,
        "n_params": n_params,
        "n_epochs": epoch,
        "best_epoch": best_epoch,
        "training_time_minutes": total_time / 60,
        "test_metrics": {
            "loss": float(test_metrics["loss"]),
            "roc_auc": float(test_metrics["roc_auc"]),
            "avg_precision": float(test_metrics["avg_precision"]),
            "accuracy": float(test_metrics["accuracy"]),
        },
        "hyperparameters": {
            "embedding_dim": embedding_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print("  - best_model.pt (model weights)")
    print("  - results.json (metrics)")
    print("  - training_curves.png")
    print("  - roc_curve.png")
    print("  - pr_curve.png")

    return model, results


if __name__ == "__main__":
    # Example usage - train a simple model
    model, results = train_model(
        model_type="embedding_mlp",
        data_dir="data/processed",
        output_dir="outputs/simple_dl",
        embedding_dim=64,
        hidden_dims=[128, 64],
        dropout=0.3,
        batch_size=512,
        learning_rate=0.001,
        n_epochs=50,
        patience=10,
    )
