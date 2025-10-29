"""
Complete example with patient features (age, sex, route).

This demonstrates how to use contextual features for better predictions.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

from data_prep_with_features import prepare_faers_data_with_features
from model_with_features import get_model_with_features


class DrugReactionDatasetWithFeatures(Dataset):
    """Dataset that includes patient features."""

    def __init__(self, data_dict: dict):
        self.drug_indices = torch.from_numpy(data_dict['drug_indices']).long()
        self.reaction_indices = torch.from_numpy(data_dict['reaction_indices']).long()
        self.labels = torch.from_numpy(data_dict['labels']).float().unsqueeze(1)

        # Optional features
        self.age = torch.from_numpy(data_dict['age']).float() if 'age' in data_dict else None
        self.sex_indices = torch.from_numpy(data_dict['sex_indices']).long() if 'sex_indices' in data_dict else None
        self.route_indices = torch.from_numpy(data_dict['route_indices']).long() if 'route_indices' in data_dict else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'drug_idx': self.drug_indices[idx],
            'reaction_idx': self.reaction_indices[idx],
            'label': self.labels[idx],
        }

        if self.age is not None:
            item['age'] = self.age[idx]
        if self.sex_indices is not None:
            item['sex_idx'] = self.sex_indices[idx]
        if self.route_indices is not None:
            item['route_idx'] = self.route_indices[idx]

        return item


def collate_fn(batch):
    """Custom collate function to handle optional features."""
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        collated[key] = torch.stack([item[key] for item in batch])

    return collated


def load_data_with_features(data_dir):
    """Load processed data with features."""
    data_dir = Path(data_dir)

    datasets = {}
    for split in ['train', 'val', 'test']:
        data_dict = dict(np.load(data_dir / f'{split}.npz'))
        datasets[split] = DrugReactionDatasetWithFeatures(data_dict)

    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    return datasets, metadata


@torch.no_grad()
def evaluate_with_features(model, dataloader, criterion, device):
    """Evaluate model with features."""
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        # Move to device
        drug_idx = batch['drug_idx'].to(device)
        reaction_idx = batch['reaction_idx'].to(device)
        labels = batch['label'].to(device)

        age = batch.get('age', None)
        if age is not None:
            age = age.to(device)

        sex_idx = batch.get('sex_idx', None)
        if sex_idx is not None:
            sex_idx = sex_idx.to(device)

        route_idx = batch.get('route_idx', None)
        if route_idx is not None:
            route_idx = route_idx.to(device)

        # Forward pass
        preds = model(drug_idx, reaction_idx, age, sex_idx, route_idx)
        loss = criterion(preds, labels)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_loss += loss.item()
        n_batches += 1

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return {
        'loss': total_loss / n_batches,
        'roc_auc': roc_auc_score(all_labels, all_preds),
        'avg_precision': average_precision_score(all_labels, all_preds),
        'accuracy': ((all_preds > 0.5) == all_labels).mean(),
    }


def train_epoch_with_features(model, dataloader, optimizer, criterion, device):
    """Train one epoch with features."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        drug_idx = batch['drug_idx'].to(device)
        reaction_idx = batch['reaction_idx'].to(device)
        labels = batch['label'].to(device)

        age = batch.get('age', None)
        if age is not None:
            age = age.to(device)

        sex_idx = batch.get('sex_idx', None)
        if sex_idx is not None:
            sex_idx = sex_idx.to(device)

        route_idx = batch.get('route_idx', None)
        if route_idx is not None:
            route_idx = route_idx.to(device)

        optimizer.zero_grad()
        preds = model(drug_idx, reaction_idx, age, sex_idx, route_idx)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    print("=" * 80)
    print("DRUG-REACTION PREDICTION WITH PATIENT FEATURES")
    print("=" * 80)
    print()

    # ==========================================================================
    # STEP 1: Create example data (REPLACE WITH YOUR ACTUAL DATA)
    # ==========================================================================
    print("Creating example data with features...")

    np.random.seed(42)

    # Example data with features
    drugs = ['ASPIRIN', 'WARFARIN', 'IBUPROFEN', 'METFORMIN', 'LISINOPRIL']
    reactions = ['bleeding', 'nausea', 'headache', 'dizziness', 'rash']
    sexes = ['M', 'F', 'UNKNOWN']
    routes = ['ORAL', 'INTRAVENOUS', 'TOPICAL', 'OTHER']

    data = []
    for _ in range(500):
        drug = np.random.choice(drugs)

        # Realistic associations
        if drug in ['ASPIRIN', 'WARFARIN']:
            reaction = np.random.choice(['bleeding', 'nausea'], p=[0.7, 0.3])
        else:
            reaction = np.random.choice(reactions)

        # Features
        age = np.random.normal(60, 15)  # Mean age 60, std 15
        age = np.clip(age, 18, 100)  # Realistic range

        sex = np.random.choice(sexes, p=[0.45, 0.45, 0.1])

        # Route depends on drug
        if drug in ['ASPIRIN', 'METFORMIN']:
            route = 'ORAL'
        elif drug == 'WARFARIN':
            route = np.random.choice(['ORAL', 'INTRAVENOUS'], p=[0.8, 0.2])
        else:
            route = np.random.choice(routes)

        data.append({
            'prod_ai': drug,
            'pt': reaction,
            'age_clean': age,
            'sex': sex,
            'route_clean': route,
        })

    df = pd.DataFrame(data)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/example_faers_with_features.csv", index=False)
    print(f"Created {len(df)} records with features\n")

    # ==========================================================================
    # REPLACE WITH YOUR DATA:
    # df = pd.read_csv("/Users/somya/Desktop/fda-reporting/data/your_data.csv")
    # ==========================================================================

    # ==========================================================================
    # STEP 2: Preprocess
    # ==========================================================================
    print("STEP 2: Preprocessing with features...")
    print("-" * 80)

    preprocessor = prepare_faers_data_with_features(
        faers_df=df,
        output_dir="data/processed_with_features",
        min_drug_occurrences=5,
        min_reaction_occurrences=5,
        use_age=True,
        use_sex=True,
        use_route=True,
    )
    print()

    # ==========================================================================
    # STEP 3: Train
    # ==========================================================================
    print("STEP 3: Training model with features...")
    print("-" * 80)

    # Load data
    datasets, metadata = load_data_with_features("data/processed_with_features")

    print(f"Features used:")
    print(f"  Age: {metadata['use_age']}")
    print(f"  Sex: {metadata['use_sex']} ({metadata['n_sex_categories']} categories)")
    print(f"  Route: {metadata['use_route']} ({metadata['n_route_categories']} categories)")
    print()

    # Create dataloaders
    train_loader = DataLoader(
        datasets['train'],
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        datasets['val'],
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        datasets['test'],
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = get_model_with_features(
        model_type="mlp",  # or "attention"
        metadata=metadata,
        embedding_dim=32,
        hidden_dims=[64, 32],
        dropout=0.2,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")

    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # Train
    best_val_auc = 0
    n_epochs = 20

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch_with_features(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_with_features(model, val_loader, criterion, device)

        scheduler.step(val_metrics['roc_auc'])

        print(f"Epoch {epoch}/{n_epochs}:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val: AUC={val_metrics['roc_auc']:.4f}, AP={val_metrics['avg_precision']:.4f}, Acc={val_metrics['accuracy']:.4f}")

        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            Path("outputs/with_features").mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'metadata': metadata,
            }, "outputs/with_features/best_model.pt")
            print("  ✓ Best model saved")

    print()

    # ==========================================================================
    # STEP 4: Evaluate
    # ==========================================================================
    print("STEP 4: Final evaluation...")
    print("-" * 80)

    test_metrics = evaluate_with_features(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  ROC AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  Average Precision: {test_metrics['avg_precision']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print()

    # ==========================================================================
    # DONE
    # ==========================================================================
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nModel with features saved to: outputs/with_features/best_model.pt")
    print("\nNow you can make predictions considering age, sex, and route!")
    print()
    print("Example usage:")
    print("  # Load model")
    print("  checkpoint = torch.load('outputs/with_features/best_model.pt')")
    print("  model.load_state_dict(checkpoint['model_state_dict'])")
    print()
    print("  # Predict with features")
    print("  # e.g., 70-year-old male taking aspirin orally")
    print("  drug_idx = drug_to_idx['ASPIRIN']")
    print("  reaction_idx = reaction_to_idx['bleeding']")
    print("  age = (70 - age_mean) / age_std  # normalized")
    print("  sex_idx = sex_to_idx['M']")
    print("  route_idx = route_to_idx['ORAL']")
    print("  prob = model(drug_idx, reaction_idx, age, sex_idx, route_idx)")


if __name__ == "__main__":
    main()
