"""
Export embeddings from trained model for t-SNE visualization.
"""

import torch
import json
import pandas as pd
from pathlib import Path
from model_with_features import get_model_with_features


def export_embeddings(
    model_path="outputs/my_model/best_model.pt", output_dir="outputs/embeddings"
):
    """Export drug and reaction embeddings to CSV."""

    print("Loading model...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]

    # Load vocabularies
    with open("data/processed_with_features/vocabularies.json") as f:
        vocab = json.load(f)

    idx_to_drug = {int(k): v for k, v in vocab["idx_to_drug"].items()}
    idx_to_reaction = {int(k): v for k, v in vocab["idx_to_reaction"].items()}

    # Load model
    model = get_model_with_features(
        model_type="mlp",
        metadata=metadata,
        embedding_dim=64,
        hidden_dims=[128, 64],
        dropout=0.3,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Model loaded: {metadata['n_drugs']} drugs, {metadata['n_reactions']} reactions"
    )

    # Extract embeddings
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Drug embeddings
    with torch.no_grad():
        drug_embeddings = model.drug_embedding.weight.cpu().numpy()

    drug_df = pd.DataFrame(
        drug_embeddings, index=[idx_to_drug[i] for i in range(len(idx_to_drug))]
    )
    drug_df.index.name = "drug"
    drug_df.to_csv(output_dir / "drug_embeddings.csv")
    print(f"Saved drug embeddings: {drug_df.shape}")

    # Reaction embeddings
    with torch.no_grad():
        reaction_embeddings = model.reaction_embedding.weight.cpu().numpy()

    reaction_df = pd.DataFrame(
        reaction_embeddings,
        index=[idx_to_reaction[i] for i in range(len(idx_to_reaction))],
    )
    reaction_df.index.name = "reaction"
    reaction_df.to_csv(output_dir / "reaction_embeddings.csv")
    print(f"Saved reaction embeddings: {reaction_df.shape}")

    print(f"\nEmbeddings saved to {output_dir}/")
    print("Now run: python visualize_embeddings.py")


if __name__ == "__main__":
    export_embeddings()
