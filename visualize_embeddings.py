"""
Visualize learned drug and reaction embeddings using t-SNE or UMAP.

This shows how the learned embeddings are better than raw t-SNE on your data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import argparse


def load_embeddings(embeddings_dir="outputs/embeddings"):
    """Load exported embeddings."""
    embeddings_dir = Path(embeddings_dir)

    drug_emb = pd.read_csv(embeddings_dir / "drug_embeddings.csv", index_col=0)
    reaction_emb = pd.read_csv(embeddings_dir / "reaction_embeddings.csv", index_col=0)

    return drug_emb, reaction_emb


def plot_tsne(
    embeddings: pd.DataFrame,
    title: str,
    output_path: str,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    annotate_top_k: int = 20,
):
    """
    Create t-SNE visualization of embeddings.

    Args:
        embeddings: DataFrame with embeddings (rows=items, cols=dimensions)
        title: Plot title
        output_path: Where to save the plot
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        random_state: Random seed
        annotate_top_k: Number of items to annotate with labels
    """
    print(f"Computing t-SNE for {len(embeddings)} items...")
    print(f"  Perplexity: {perplexity}")
    print(f"  Iterations: {n_iter}")

    # Compute t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1,
    )
    coords = tsne.fit_transform(embeddings.values)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Scatter plot
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        alpha=0.6,
        s=100,
        c=range(len(coords)),
        cmap="viridis",
        edgecolors="black",
        linewidth=0.5,
    )

    # Annotate some points
    if annotate_top_k > 0:
        # Annotate items that are furthest from origin (most distinct)
        distances = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
        top_indices = np.argsort(distances)[-annotate_top_k:]

        for idx in top_indices:
            label = embeddings.index[idx]
            ax.annotate(
                label,
                (coords[idx, 0], coords[idx, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, label="Index", ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close()


def plot_umap(
    embeddings: pd.DataFrame,
    title: str,
    output_path: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    annotate_top_k: int = 20,
):
    """
    Create UMAP visualization of embeddings.

    Requires: pip install umap-learn

    Args:
        embeddings: DataFrame with embeddings
        title: Plot title
        output_path: Where to save
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed
        annotate_top_k: Number of items to label
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return

    print(f"Computing UMAP for {len(embeddings)} items...")
    print(f"  n_neighbors: {n_neighbors}")
    print(f"  min_dist: {min_dist}")

    # Compute UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=True,
    )
    coords = reducer.fit_transform(embeddings.values)

    # Create plot (same as t-SNE)
    fig, ax = plt.subplots(figsize=(14, 10))

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        alpha=0.6,
        s=100,
        c=range(len(coords)),
        cmap="viridis",
        edgecolors="black",
        linewidth=0.5,
    )

    if annotate_top_k > 0:
        distances = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
        top_indices = np.argsort(distances)[-annotate_top_k:]

        for idx in top_indices:
            label = embeddings.index[idx]
            ax.annotate(
                label,
                (coords[idx, 0], coords[idx, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, label="Index", ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize learned drug and reaction embeddings"
    )
    parser.add_argument(
        "--embeddings_dir",
        default="outputs/embeddings",
        help="Directory with exported embeddings",
    )
    parser.add_argument(
        "--output_dir", default="outputs/visualizations", help="Where to save plots"
    )
    parser.add_argument(
        "--method",
        choices=["tsne", "umap", "both"],
        default="both",
        help="Visualization method",
    )
    parser.add_argument(
        "--perplexity", type=int, default=30, help="t-SNE perplexity (5-50)"
    )
    parser.add_argument(
        "--annotate", type=int, default=20, help="Number of items to label"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print("Loading embeddings...")
    drug_emb, reaction_emb = load_embeddings(args.embeddings_dir)
    print(f"Loaded {len(drug_emb)} drug embeddings (dim={drug_emb.shape[1]})")
    print(
        f"Loaded {len(reaction_emb)} reaction embeddings (dim={reaction_emb.shape[1]})"
    )
    print()

    # Visualize drugs
    if args.method in ["tsne", "both"]:
        print("=" * 60)
        print("Creating t-SNE visualization for drugs...")
        print("=" * 60)
        plot_tsne(
            drug_emb,
            title="Drug Embeddings (t-SNE)",
            output_path=output_dir / "drugs_tsne.png",
            perplexity=min(args.perplexity, len(drug_emb) - 1),
            annotate_top_k=args.annotate,
        )
        print()

        print("=" * 60)
        print("Creating t-SNE visualization for reactions...")
        print("=" * 60)
        plot_tsne(
            reaction_emb,
            title="Reaction Embeddings (t-SNE)",
            output_path=output_dir / "reactions_tsne.png",
            perplexity=min(args.perplexity, len(reaction_emb) - 1),
            annotate_top_k=args.annotate,
        )
        print()

    if args.method in ["umap", "both"]:
        print("=" * 60)
        print("Creating UMAP visualization for drugs...")
        print("=" * 60)
        plot_umap(
            drug_emb,
            title="Drug Embeddings (UMAP)",
            output_path=output_dir / "drugs_umap.png",
            annotate_top_k=args.annotate,
        )
        print()

        print("=" * 60)
        print("Creating UMAP visualization for reactions...")
        print("=" * 60)
        plot_umap(
            reaction_emb,
            title="Reaction Embeddings (UMAP)",
            output_path=output_dir / "reactions_umap.png",
            annotate_top_k=args.annotate,
        )
        print()

    print("=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nVisualizations saved to {output_dir}/")
    print("\nFiles created:")
    if args.method in ["tsne", "both"]:
        print("  - drugs_tsne.png")
        print("  - reactions_tsne.png")
    if args.method in ["umap", "both"]:
        print("  - drugs_umap.png")
        print("  - reactions_umap.png")


if __name__ == "__main__":
    # Can run with command line args or directly
    import sys

    if len(sys.argv) == 1:
        # No args - run with defaults
        print("Running with default settings...")
        print("Usage: python visualize_embeddings.py --method tsne --perplexity 30")
        print()
        sys.argv = ["visualize_embeddings.py"]  # Reset for argparse

    main()
