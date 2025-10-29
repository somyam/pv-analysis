"""
Simple script to train model with YOUR FAERS data using existing functions.
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Import your existing functions
from data_prep_with_features import prepare_faers_data_with_features
from model_with_features import get_model_with_features


# =============================================================================
# CONFIGURATION
# =============================================================================

# UPDATE THIS PATH TO YOUR DATA
DATA_PATH = "/Users/somya/Desktop/malade/simple_dl/faers_data.csv"

# Data preprocessing settings
MIN_DRUG_OCCURRENCES = 10
MIN_REACTION_OCCURRENCES = 10
USE_AGE = True
USE_SEX = True
USE_ROUTE = True

# Model settings
MODEL_TYPE = "mlp"  # or "attention"
EMBEDDING_DIM = 64
HIDDEN_DIMS = [128, 64]
DROPOUT = 0.3

# Training settings
BATCH_SIZE = 3000
LEARNING_RATE = 0.01  # Higher for SGD
WEIGHT_DECAY = 1e-5
N_EPOCHS = 30  # Fewer epochs
PATIENCE = 5  # Stop earlier if not improving

OUTPUT_DIR = "outputs/my_model"


# =============================================================================
# DATASET CLASS
# =============================================================================


class FAERSDataset(Dataset):
    """Dataset for drug-reaction data with features."""

    def __init__(self, data_dict):
        self.drug_indices = torch.from_numpy(data_dict["drug_indices"]).long()
        self.reaction_indices = torch.from_numpy(data_dict["reaction_indices"]).long()
        self.labels = torch.from_numpy(data_dict["labels"]).float().unsqueeze(1)

        # Optional features
        self.age = (
            torch.from_numpy(data_dict["age"]).float() if "age" in data_dict else None
        )
        self.sex_indices = (
            torch.from_numpy(data_dict["sex_indices"]).long()
            if "sex_indices" in data_dict
            else None
        )
        self.route_indices = (
            torch.from_numpy(data_dict["route_indices"]).long()
            if "route_indices" in data_dict
            else None
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "drug_idx": self.drug_indices[idx],
            "reaction_idx": self.reaction_indices[idx],
            "label": self.labels[idx],
        }

        if self.age is not None:
            item["age"] = self.age[idx]
        if self.sex_indices is not None:
            item["sex_idx"] = self.sex_indices[idx]
        if self.route_indices is not None:
            item["route_idx"] = self.route_indices[idx]

        return item


def collate_fn(batch):
    """Collate function for batching."""
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch]) for key in keys}


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in loader:
        # Move to device
        drug_idx = batch["drug_idx"].to(device)
        reaction_idx = batch["reaction_idx"].to(device)
        labels = batch["label"].to(device)

        age = batch.get("age")
        if age is not None:
            age = age.to(device)

        sex_idx = batch.get("sex_idx")
        if sex_idx is not None:
            sex_idx = sex_idx.to(device)

        route_idx = batch.get("route_idx")
        if route_idx is not None:
            route_idx = route_idx.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(drug_idx, reaction_idx, age, sex_idx, route_idx)
        loss = criterion(predictions, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0

    for batch in loader:
        drug_idx = batch["drug_idx"].to(device)
        reaction_idx = batch["reaction_idx"].to(device)
        labels = batch["label"].to(device)

        age = batch.get("age")
        if age is not None:
            age = age.to(device)

        sex_idx = batch.get("sex_idx")
        if sex_idx is not None:
            sex_idx = sex_idx.to(device)

        route_idx = batch.get("route_idx")
        if route_idx is not None:
            route_idx = route_idx.to(device)

        predictions = model(drug_idx, reaction_idx, age, sex_idx, route_idx)
        loss = criterion(predictions, labels)

        all_predictions.append(predictions.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_loss += loss.item()

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    return {
        "loss": total_loss / len(loader),
        "roc_auc": roc_auc_score(all_labels, all_predictions),
        "avg_precision": average_precision_score(all_labels, all_predictions),
        "accuracy": ((all_predictions > 0.5) == all_labels).mean(),
    }


# =============================================================================
# MAIN SCRIPT
# =============================================================================


def main():
    print("=" * 80)
    print("TRAINING MODEL WITH YOUR FAERS DATA")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # STEP 1: Load data
    # -------------------------------------------------------------------------
    print("STEP 1: Loading data...")
    print("-" * 80)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {df.columns.tolist()}")
    print()

    # -------------------------------------------------------------------------
    # STEP 2: Preprocess with features
    # -------------------------------------------------------------------------
    print("STEP 2: Preprocessing with features...")
    print("-" * 80)

    prepare_faers_data_with_features(
        faers_df=df,
        output_dir="data/processed_with_features",
        min_drug_occurrences=MIN_DRUG_OCCURRENCES,
        min_reaction_occurrences=MIN_REACTION_OCCURRENCES,
        use_age=USE_AGE,
        use_sex=USE_SEX,
        use_route=USE_ROUTE,
    )
    print()

    # -------------------------------------------------------------------------
    # STEP 3: Load processed data
    # -------------------------------------------------------------------------
    print("STEP 3: Loading processed data...")
    print("-" * 80)

    data_dir = Path("data/processed_with_features")

    datasets = {}
    for split in ["train", "val", "test"]:
        data_dict = dict(np.load(data_dir / f"{split}.npz"))
        datasets[split] = FAERSDataset(data_dict)

    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    print(f"Drugs: {metadata['n_drugs']}")
    print(f"Reactions: {metadata['n_reactions']}")
    print(
        f"Features - Age: {metadata['use_age']}, Sex: {metadata['use_sex']}, Route: {metadata['use_route']}"
    )
    print(
        f"Train: {len(datasets['train']):,}, Val: {len(datasets['val']):,}, Test: {len(datasets['test']):,}"
    )
    print()

    # -------------------------------------------------------------------------
    # STEP 4: Create model
    # -------------------------------------------------------------------------
    print("STEP 4: Creating model...")
    print("-" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = get_model_with_features(
        model_type=MODEL_TYPE,
        metadata=metadata,
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print()

    # -------------------------------------------------------------------------
    # STEP 5: Training
    # -------------------------------------------------------------------------
    print("STEP 5: Training...")
    print("-" * 80)

    train_loader = DataLoader(
        datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)

    best_val_auc = 0
    patience_counter = 0

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["roc_auc"])

        print(
            f"Epoch {epoch:2d}/{N_EPOCHS}: "
            f"train_loss={train_loss:.4f}, "
            f"val_auc={val_metrics['roc_auc']:.4f}, "
            f"val_ap={val_metrics['avg_precision']:.4f}"
        )

        # Save best model
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            patience_counter = 0

            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": metadata,
                    "val_metrics": val_metrics,
                },
                Path(OUTPUT_DIR) / "best_model.pt",
            )
            print(f"  ✓ Best model saved (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print()

    # -------------------------------------------------------------------------
    # STEP 6: Test evaluation
    # -------------------------------------------------------------------------
    print("STEP 6: Final evaluation...")
    print("-" * 80)

    checkpoint = torch.load(Path(OUTPUT_DIR) / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  ROC AUC:           {test_metrics['roc_auc']:.4f}")
    print(f"  Average Precision: {test_metrics['avg_precision']:.4f}")
    print(f"  Accuracy:          {test_metrics['accuracy']:.4f}")

    # Save results
    results = {
        "test_metrics": {
            k: float(v)
            for k, v in test_metrics.items()
            if k not in ["predictions", "labels"]
        },
        "metadata": metadata,
        "config": {
            "model_type": MODEL_TYPE,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dims": HIDDEN_DIMS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        },
    }

    with open(Path(OUTPUT_DIR) / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nModel saved to: {OUTPUT_DIR}/best_model.pt")
    print(f"Results saved to: {OUTPUT_DIR}/results.json")
    print()


if __name__ == "__main__":
    main()
