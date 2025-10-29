"""
Compare drugs after training is complete.
"""

import torch
import json
import numpy as np
from pathlib import Path
from model_with_features import get_model_with_features


class DrugComparator:
    """Compare drug-reaction associations."""

    def __init__(self, model_path="outputs/my_model/best_model.pt"):
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.metadata = checkpoint['metadata']

        # Load vocabularies
        with open("data/processed_with_features/vocabularies.json") as f:
            vocab = json.load(f)

        self.drug_to_idx = vocab['drug_to_idx']
        self.reaction_to_idx = vocab['reaction_to_idx']
        self.idx_to_drug = {int(k): v for k, v in vocab['idx_to_drug'].items()}
        self.idx_to_reaction = {int(k): v for k, v in vocab['idx_to_reaction'].items()}

        if 'sex_to_idx' in vocab:
            self.sex_to_idx = vocab['sex_to_idx']
        if 'route_to_idx' in vocab:
            self.route_to_idx = vocab['route_to_idx']

        # Load model
        self.model = get_model_with_features(
            model_type="mlp",
            metadata=self.metadata,
            embedding_dim=64,
            hidden_dims=[128, 64],
            dropout=0.3,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded from {model_path}")
        print(f"Vocabulary: {len(self.drug_to_idx)} drugs, {len(self.reaction_to_idx)} reactions")

    def predict(self, drug, reaction, age=None, sex=None, route=None):
        """Predict association probability."""
        drug = drug.upper().strip()
        reaction = reaction.lower().strip()

        if drug not in self.drug_to_idx:
            raise ValueError(f"Drug '{drug}' not in vocabulary")
        if reaction not in self.reaction_to_idx:
            raise ValueError(f"Reaction '{reaction}' not in vocabulary")

        drug_idx = torch.tensor([self.drug_to_idx[drug]])
        reaction_idx = torch.tensor([self.reaction_to_idx[reaction]])

        # Handle features
        age_tensor = None
        if age is not None and self.metadata['use_age']:
            age_norm = (age - self.metadata['age_mean']) / self.metadata['age_std']
            age_tensor = torch.tensor([age_norm], dtype=torch.float32)

        sex_idx = None
        if sex is not None and self.metadata['use_sex']:
            sex = sex.upper()
            if sex in self.sex_to_idx:
                sex_idx = torch.tensor([self.sex_to_idx[sex]])

        route_idx = None
        if route is not None and self.metadata['use_route']:
            route = route.upper()
            if route in self.route_to_idx:
                route_idx = torch.tensor([self.route_to_idx[route]])

        with torch.no_grad():
            prob = self.model(drug_idx, reaction_idx, age_tensor, sex_idx, route_idx)

        return prob.item()

    def compare_drugs(self, drugs, reaction, age=None, sex=None, route=None):
        """Compare multiple drugs for same reaction."""
        results = []
        for drug in drugs:
            try:
                prob = self.predict(drug, reaction, age, sex, route)
                results.append((drug, prob))
            except ValueError as e:
                print(f"Skipping {drug}: {e}")

        # Sort by probability (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def top_reactions_for_drug(self, drug, top_k=10, age=None, sex=None, route=None):
        """Get top reactions for a drug."""
        drug = drug.upper().strip()
        if drug not in self.drug_to_idx:
            raise ValueError(f"Drug '{drug}' not in vocabulary")

        drug_idx = self.drug_to_idx[drug]
        n_reactions = len(self.reaction_to_idx)

        # Prepare tensors
        drug_indices = torch.tensor([drug_idx] * n_reactions)
        reaction_indices = torch.arange(n_reactions)

        age_tensor = None
        if age is not None and self.metadata['use_age']:
            age_norm = (age - self.metadata['age_mean']) / self.metadata['age_std']
            age_tensor = torch.tensor([age_norm] * n_reactions, dtype=torch.float32)

        sex_idx = None
        if sex is not None and self.metadata['use_sex']:
            sex = sex.upper()
            if sex in self.sex_to_idx:
                sex_idx = torch.tensor([self.sex_to_idx[sex]] * n_reactions)

        route_idx = None
        if route is not None and self.metadata['use_route']:
            route = route.upper()
            if route in self.route_to_idx:
                route_idx = torch.tensor([self.route_to_idx[route]] * n_reactions)

        with torch.no_grad():
            probs = self.model(drug_indices, reaction_indices, age_tensor, sex_idx, route_idx)

        probs = probs.numpy().squeeze()
        top_indices = np.argsort(probs)[-top_k:][::-1]

        return [(self.idx_to_reaction[int(idx)], float(probs[idx])) for idx in top_indices]


if __name__ == "__main__":
    print("Loading model...")
    comparator = DrugComparator("outputs/my_model/best_model.pt")

    print("\n" + "="*80)
    print("DRUG COMPARISON EXAMPLES")
    print("="*80)

    # Example 1: Compare drugs for bleeding risk
    print("\n1. Compare drugs for BLEEDING risk:")
    print("-"*80)
    drugs = ["WARFARIN", "ASPIRIN", "IBUPROFEN"]
    results = comparator.compare_drugs(drugs, "bleeding")
    for i, (drug, prob) in enumerate(results, 1):
        print(f"  {i}. {drug}: {prob:.4f}")

    # Example 2: Top reactions for a drug
    print("\n2. Top reactions for WARFARIN:")
    print("-"*80)
    try:
        top = comparator.top_reactions_for_drug("WARFARIN", top_k=10)
        for i, (reaction, prob) in enumerate(top, 1):
            print(f"  {i}. {reaction}: {prob:.4f}")
    except ValueError as e:
        print(f"  {e}")

    # Example 3: With patient features
    print("\n3. WARFARIN bleeding risk by patient age:")
    print("-"*80)
    for age in [30, 50, 70, 90]:
        try:
            prob = comparator.predict("WARFARIN", "bleeding", age=age, sex="M", route="ORAL")
            print(f"  Age {age}: {prob:.4f}")
        except ValueError as e:
            print(f"  {e}")

    print("\n" + "="*80)
    print("Create your own comparisons!")
    print("="*80)
    print("\nExamples:")
    print("  # Compare drugs")
    print("  results = comparator.compare_drugs(['DRUG1', 'DRUG2'], 'reaction')")
    print()
    print("  # Top reactions")
    print("  top = comparator.top_reactions_for_drug('DRUG_NAME', top_k=10)")
    print()
    print("  # With features")
    print("  prob = comparator.predict('DRUG', 'reaction', age=70, sex='M', route='ORAL')")
    print()
