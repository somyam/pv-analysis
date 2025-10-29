"""
Neural network models with patient features (age, sex, route).

These models combine drug/reaction embeddings with contextual features
for improved prediction accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class DrugReactionModelWithFeatures(nn.Module):
    """
    Embedding-based model with patient/drug features.

    Architecture:
    1. Embed drugs and reactions
    2. Embed categorical features (sex, route)
    3. Combine all features
    4. MLP for prediction
    """

    def __init__(
        self,
        n_drugs: int,
        n_reactions: int,
        embedding_dim: int = 64,
        # Feature dimensions
        n_sex_categories: int = 0,
        n_route_categories: int = 0,
        sex_embedding_dim: int = 8,
        route_embedding_dim: int = 16,
        use_age: bool = False,
        # MLP configuration
        hidden_dims: Optional[list] = None,
        dropout: float = 0.3,
    ):
        """
        Args:
            n_drugs: Number of unique drugs
            n_reactions: Number of unique reactions
            embedding_dim: Dimensionality of drug/reaction embeddings
            n_sex_categories: Number of sex categories (0 = don't use)
            n_route_categories: Number of administration routes (0 = don't use)
            sex_embedding_dim: Dimension for sex embeddings
            route_embedding_dim: Dimension for route embeddings
            use_age: Whether age is provided as feature
            hidden_dims: Hidden layer sizes
            dropout: Dropout probability
        """
        super().__init__()

        self.n_drugs = n_drugs
        self.n_reactions = n_reactions
        self.embedding_dim = embedding_dim
        self.use_age = use_age
        self.use_sex = n_sex_categories > 0
        self.use_route = n_route_categories > 0

        # Drug and reaction embeddings
        self.drug_embedding = nn.Embedding(n_drugs, embedding_dim)
        self.reaction_embedding = nn.Embedding(n_reactions, embedding_dim)

        # Initialize embeddings
        nn.init.normal_(self.drug_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.reaction_embedding.weight, mean=0, std=0.01)

        # Feature embeddings
        if self.use_sex:
            self.sex_embedding = nn.Embedding(n_sex_categories, sex_embedding_dim)
            nn.init.normal_(self.sex_embedding.weight, mean=0, std=0.01)

        if self.use_route:
            self.route_embedding = nn.Embedding(n_route_categories, route_embedding_dim)
            nn.init.normal_(self.route_embedding.weight, mean=0, std=0.01)

        # Calculate total input dimension for MLP
        input_dim = embedding_dim * 2  # drug + reaction
        if self.use_age:
            input_dim += 1  # normalized age
        if self.use_sex:
            input_dim += sex_embedding_dim
        if self.use_route:
            input_dim += route_embedding_dim

        # MLP layers
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        drug_indices: torch.Tensor,
        reaction_indices: torch.Tensor,
        age: Optional[torch.Tensor] = None,
        sex_indices: Optional[torch.Tensor] = None,
        route_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            drug_indices: (batch_size,) with drug indices
            reaction_indices: (batch_size,) with reaction indices
            age: Optional (batch_size,) with normalized ages
            sex_indices: Optional (batch_size,) with sex indices
            route_indices: Optional (batch_size,) with route indices

        Returns:
            (batch_size, 1) predicted probabilities
        """
        batch_size = drug_indices.size(0)

        # Get drug and reaction embeddings
        drug_emb = self.drug_embedding(drug_indices)  # (batch, embedding_dim)
        reaction_emb = self.reaction_embedding(reaction_indices)  # (batch, embedding_dim)

        # Start with drug + reaction
        features = [drug_emb, reaction_emb]

        # Add age if available
        if self.use_age and age is not None:
            age_feature = age.view(batch_size, 1)  # (batch, 1)
            features.append(age_feature)

        # Add sex embedding if available
        if self.use_sex and sex_indices is not None:
            sex_emb = self.sex_embedding(sex_indices)  # (batch, sex_emb_dim)
            features.append(sex_emb)

        # Add route embedding if available
        if self.use_route and route_indices is not None:
            route_emb = self.route_embedding(route_indices)  # (batch, route_emb_dim)
            features.append(route_emb)

        # Concatenate all features
        combined = torch.cat(features, dim=1)

        # Pass through MLP
        output = self.mlp(combined)

        return torch.sigmoid(output)

    def get_drug_embedding(self, drug_idx: int) -> torch.Tensor:
        """Get embedding for a specific drug."""
        return self.drug_embedding.weight[drug_idx]

    def get_reaction_embedding(self, reaction_idx: int) -> torch.Tensor:
        """Get embedding for a specific reaction."""
        return self.reaction_embedding.weight[reaction_idx]


class AttentionModelWithFeatures(nn.Module):
    """
    Advanced model with attention mechanism over features.

    This learns to weight different features based on the drug-reaction pair.
    """

    def __init__(
        self,
        n_drugs: int,
        n_reactions: int,
        embedding_dim: int = 64,
        n_sex_categories: int = 0,
        n_route_categories: int = 0,
        sex_embedding_dim: int = 8,
        route_embedding_dim: int = 16,
        use_age: bool = False,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        """
        Args:
            Similar to DrugReactionModelWithFeatures
            hidden_dim: Hidden dimension for attention and MLP
        """
        super().__init__()

        self.use_age = use_age
        self.use_sex = n_sex_categories > 0
        self.use_route = n_route_categories > 0

        # Embeddings
        self.drug_embedding = nn.Embedding(n_drugs, embedding_dim)
        self.reaction_embedding = nn.Embedding(n_reactions, embedding_dim)

        if self.use_sex:
            self.sex_embedding = nn.Embedding(n_sex_categories, sex_embedding_dim)

        if self.use_route:
            self.route_embedding = nn.Embedding(n_route_categories, route_embedding_dim)

        # Feature projection to common dimension
        self.drug_proj = nn.Linear(embedding_dim, hidden_dim)
        self.reaction_proj = nn.Linear(embedding_dim, hidden_dim)

        if self.use_age:
            self.age_proj = nn.Linear(1, hidden_dim)

        if self.use_sex:
            self.sex_proj = nn.Linear(sex_embedding_dim, hidden_dim)

        if self.use_route:
            self.route_proj = nn.Linear(route_embedding_dim, hidden_dim)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Final prediction
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        drug_indices: torch.Tensor,
        reaction_indices: torch.Tensor,
        age: Optional[torch.Tensor] = None,
        sex_indices: Optional[torch.Tensor] = None,
        route_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with attention over features."""
        batch_size = drug_indices.size(0)

        # Get embeddings and project to common dimension
        drug_emb = self.drug_embedding(drug_indices)
        drug_feat = self.drug_proj(drug_emb).unsqueeze(1)  # (batch, 1, hidden)

        reaction_emb = self.reaction_embedding(reaction_indices)
        reaction_feat = self.reaction_proj(reaction_emb).unsqueeze(1)

        # Collect features
        features = [drug_feat, reaction_feat]  # List of (batch, 1, hidden)

        if self.use_age and age is not None:
            age_feat = self.age_proj(age.view(batch_size, 1)).unsqueeze(1)
            features.append(age_feat)

        if self.use_sex and sex_indices is not None:
            sex_emb = self.sex_embedding(sex_indices)
            sex_feat = self.sex_proj(sex_emb).unsqueeze(1)
            features.append(sex_feat)

        if self.use_route and route_indices is not None:
            route_emb = self.route_embedding(route_indices)
            route_feat = self.route_proj(route_emb).unsqueeze(1)
            features.append(route_feat)

        # Stack features: (batch, n_features, hidden)
        stacked_features = torch.cat(features, dim=1)

        # Compute attention weights
        attention_scores = self.attention(stacked_features)  # (batch, n_features, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # Normalize over features

        # Weighted sum of features
        weighted_features = (stacked_features * attention_weights).sum(dim=1)  # (batch, hidden)

        # Final prediction
        output = self.predictor(weighted_features)

        return torch.sigmoid(output)


def get_model_with_features(
    model_type: str,
    metadata: Dict,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models with features.

    Args:
        model_type: "mlp" or "attention"
        metadata: Dict with n_drugs, n_reactions, use_age, use_sex, use_route, etc.
        **kwargs: Additional model arguments

    Returns:
        PyTorch model
    """
    model_args = {
        'n_drugs': metadata['n_drugs'],
        'n_reactions': metadata['n_reactions'],
        'n_sex_categories': metadata.get('n_sex_categories', 0),
        'n_route_categories': metadata.get('n_route_categories', 0),
        'use_age': metadata.get('use_age', False),
    }
    model_args.update(kwargs)

    if model_type == "mlp":
        return DrugReactionModelWithFeatures(**model_args)
    elif model_type == "attention":
        return AttentionModelWithFeatures(**model_args)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    print("Example: Create model with features\n")

    # Example metadata
    metadata = {
        'n_drugs': 100,
        'n_reactions': 200,
        'n_sex_categories': 3,  # M, F, UNKNOWN
        'n_route_categories': 21,  # 20 top routes + OTHER
        'use_age': True,
        'use_sex': True,
        'use_route': True,
    }

    # Create MLP model
    model = get_model_with_features(
        "mlp",
        metadata=metadata,
        embedding_dim=64,
        hidden_dims=[128, 64],
        dropout=0.3,
    )

    # Test forward pass
    batch_size = 32
    drug_idx = torch.randint(0, 100, (batch_size,))
    reaction_idx = torch.randint(0, 200, (batch_size,))
    age = torch.randn(batch_size)  # Normalized age
    sex_idx = torch.randint(0, 3, (batch_size,))
    route_idx = torch.randint(0, 21, (batch_size,))

    output = model(drug_idx, reaction_idx, age, sex_idx, route_idx)
    print(f"Model output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
