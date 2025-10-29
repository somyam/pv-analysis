"""
Enhanced data preparation with patient features (age, sex, route).

This extends the basic data_prep.py to include contextual features
that can improve prediction accuracy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import json


class FAERSDataPreprocessorWithFeatures:
    """Preprocess FAERS data including patient and drug features."""

    def __init__(
        self,
        min_drug_occurrences: int = 10,
        min_reaction_occurrences: int = 10,
        test_split: float = 0.2,
        val_split: float = 0.1,
        use_age: bool = True,
        use_sex: bool = True,
        use_route: bool = True,
    ):
        """
        Args:
            min_drug_occurrences: Minimum times a drug must appear
            min_reaction_occurrences: Minimum times a reaction must appear
            test_split: Fraction for testing
            val_split: Fraction of training for validation
            use_age: Whether to use age as a feature
            use_sex: Whether to use sex as a feature
            use_route: Whether to use administration route as a feature
        """
        self.min_drug_occurrences = min_drug_occurrences
        self.min_reaction_occurrences = min_reaction_occurrences
        self.test_split = test_split
        self.val_split = val_split
        self.use_age = use_age
        self.use_sex = use_sex
        self.use_route = use_route

        # Vocabularies
        self.drug_to_idx = {}
        self.reaction_to_idx = {}
        self.sex_to_idx = {}
        self.route_to_idx = {}

        self.idx_to_drug = {}
        self.idx_to_reaction = {}
        self.idx_to_sex = {}
        self.idx_to_route = {}

        # Feature statistics
        self.age_mean = 0
        self.age_std = 1

    def load_and_filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load FAERS data with features and filter by occurrence thresholds.

        Args:
            df: DataFrame with columns ['drugname' or 'prod_ai', 'pt',
                'age_clean', 'sex', 'route_clean']

        Returns:
            Filtered DataFrame
        """
        # Determine column names
        drug_col = "prod_ai" if "prod_ai" in df.columns else "drugname"
        reaction_col = "pt"

        # Required columns
        required = [drug_col, reaction_col]
        available_features = []

        if self.use_age and "age_clean" in df.columns:
            required.append("age_clean")
            available_features.append("age_clean")

        if self.use_sex and "sex" in df.columns:
            required.append("sex")
            available_features.append("sex")

        if self.use_route and "route_clean" in df.columns:
            required.append("route_clean")
            available_features.append("route_clean")

        print(f"Original data: {len(df)} records")
        print(f"Using features: {available_features}")

        # Clean drug and reaction names
        df[drug_col] = df[drug_col].astype(str).str.upper().str.strip()
        df[reaction_col] = df[reaction_col].astype(str).str.lower().str.strip()

        # Remove nulls in required columns
        df_clean = df[required].copy()
        df_clean = df_clean.dropna()
        print(f"After removing nulls: {len(df_clean)} records")

        # Filter by drug/reaction occurrence
        drug_counts = df_clean[drug_col].value_counts()
        reaction_counts = df_clean[reaction_col].value_counts()

        valid_drugs = drug_counts[drug_counts >= self.min_drug_occurrences].index
        valid_reactions = reaction_counts[
            reaction_counts >= self.min_reaction_occurrences
        ].index

        df_filtered = df_clean[
            df_clean[drug_col].isin(valid_drugs)
            & df_clean[reaction_col].isin(valid_reactions)
        ].copy()

        print(f"After filtering rare drugs/reactions: {len(df_filtered)} records")

        # Standardize column names
        rename_map = {drug_col: "drug", reaction_col: "reaction"}
        df_filtered = df_filtered.rename(columns=rename_map)

        # Clean age
        if "age_clean" in df_filtered.columns:
            df_filtered["age_clean"] = pd.to_numeric(
                df_filtered["age_clean"], errors="coerce"
            )
            # Remove unrealistic ages
            df_filtered = df_filtered[
                (df_filtered["age_clean"] >= 0) & (df_filtered["age_clean"] <= 120)
            ]
            # Compute statistics for normalization
            self.age_mean = df_filtered["age_clean"].mean()
            self.age_std = df_filtered["age_clean"].std()
            print(f"Age: mean={self.age_mean:.1f}, std={self.age_std:.1f}")

        # Clean sex
        if "sex" in df_filtered.columns:
            df_filtered["sex"] = df_filtered["sex"].astype(str).str.upper().str.strip()
            # Keep only M, F, UNKNOWN
            df_filtered.loc[~df_filtered["sex"].isin(["M", "F"]), "sex"] = "UNKNOWN"
            print(f"Sex distribution: {df_filtered['sex'].value_counts().to_dict()}")

        # Clean route
        if "route_clean" in df_filtered.columns:
            df_filtered["route_clean"] = (
                df_filtered["route_clean"].astype(str).str.upper().str.strip()
            )
            # Keep top N routes, rest as OTHER
            route_counts = df_filtered["route_clean"].value_counts()
            top_routes = route_counts.head(20).index  # Keep top 20 routes
            df_filtered.loc[
                ~df_filtered["route_clean"].isin(top_routes), "route_clean"
            ] = "OTHER"
            print(f"Routes: {df_filtered['route_clean'].nunique()} categories")

        print(f"\nFinal dataset: {len(df_filtered)} records")
        print(f"Unique drugs: {df_filtered['drug'].nunique()}")
        print(f"Unique reactions: {df_filtered['reaction'].nunique()}")

        return df_filtered

    def create_vocabularies(self, df: pd.DataFrame):
        """Create vocabularies for all categorical features."""
        # Drug and reaction vocabularies
        unique_drugs = sorted(df["drug"].unique())
        unique_reactions = sorted(df["reaction"].unique())

        self.drug_to_idx = {drug: idx for idx, drug in enumerate(unique_drugs)}
        self.reaction_to_idx = {
            reaction: idx for idx, reaction in enumerate(unique_reactions)
        }

        self.idx_to_drug = {idx: drug for drug, idx in self.drug_to_idx.items()}
        self.idx_to_reaction = {
            idx: reaction for reaction, idx in self.reaction_to_idx.items()
        }

        # Sex vocabulary
        if "sex" in df.columns:
            unique_sex = sorted(df["sex"].unique())
            self.sex_to_idx = {sex: idx for idx, sex in enumerate(unique_sex)}
            self.idx_to_sex = {idx: sex for sex, idx in self.sex_to_idx.items()}

        # Route vocabulary
        if "route_clean" in df.columns:
            unique_routes = sorted(df["route_clean"].unique())
            self.route_to_idx = {route: idx for idx, route in enumerate(unique_routes)}
            self.idx_to_route = {idx: route for route, idx in self.route_to_idx.items()}

        print(f"\nVocabulary sizes:")
        print(f"  Drugs: {len(self.drug_to_idx)}")
        print(f"  Reactions: {len(self.reaction_to_idx)}")
        if self.sex_to_idx:
            print(f"  Sex: {len(self.sex_to_idx)}")
        if self.route_to_idx:
            print(f"  Routes: {len(self.route_to_idx)}")

    def create_training_data_with_features(
        self,
        df: pd.DataFrame,
        negative_sampling_ratio: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Create training data with features.

        Returns dict with:
            - drug_indices
            - reaction_indices
            - labels
            - age (optional, normalized)
            - sex_indices (optional)
            - route_indices (optional)
        """
        # Encode categorical features
        df["drug_idx"] = df["drug"].map(self.drug_to_idx)
        df["reaction_idx"] = df["reaction"].map(self.reaction_to_idx)

        if "sex" in df.columns:
            df["sex_idx"] = df["sex"].map(self.sex_to_idx)

        if "route_clean" in df.columns:
            df["route_idx"] = df["route_clean"].map(self.route_to_idx)

        if "age_clean" in df.columns:
            # Normalize age
            df["age_norm"] = (df["age_clean"] - self.age_mean) / self.age_std

        # Positive examples (observed associations)
        pos_data = {
            "drug_indices": df["drug_idx"].values,
            "reaction_indices": df["reaction_idx"].values,
            "labels": np.ones(len(df), dtype=np.float32),
        }

        if "age_norm" in df.columns:
            pos_data["age"] = df["age_norm"].values.astype(np.float32)

        if "sex_idx" in df.columns:
            pos_data["sex_indices"] = df["sex_idx"].values.astype(np.int64)

        if "route_idx" in df.columns:
            pos_data["route_indices"] = df["route_idx"].values.astype(np.int64)

        n_positives = len(df)
        n_negatives = int(n_positives * negative_sampling_ratio)

        print(f"\nCreating training data:")
        print(f"  Positive examples: {n_positives}")
        print(f"  Negative examples: {n_negatives}")

        # Negative examples (random drug-reaction pairs not in data)
        # Sample drugs and reactions
        neg_drug_idx = np.random.randint(0, len(self.drug_to_idx), n_negatives)
        neg_reaction_idx = np.random.randint(0, len(self.reaction_to_idx), n_negatives)

        neg_data = {
            "drug_indices": neg_drug_idx,
            "reaction_indices": neg_reaction_idx,
            "labels": np.zeros(n_negatives, dtype=np.float32),
        }

        # For negative examples, sample features from the distribution
        if "age_norm" in df.columns:
            # Sample ages from observed distribution
            neg_data["age"] = np.random.choice(
                df["age_norm"].values, n_negatives
            ).astype(np.float32)

        if "sex_idx" in df.columns:
            # Sample sex from observed distribution
            neg_data["sex_indices"] = np.random.choice(
                df["sex_idx"].values, n_negatives
            ).astype(np.int64)

        if "route_idx" in df.columns:
            # Sample routes from observed distribution
            neg_data["route_indices"] = np.random.choice(
                df["route_idx"].values, n_negatives
            ).astype(np.int64)

        # Combine positive and negative
        combined_data = {}
        for key in pos_data.keys():
            combined_data[key] = np.concatenate([pos_data[key], neg_data[key]])

        # Shuffle
        shuffle_idx = np.random.permutation(len(combined_data["labels"]))
        for key in combined_data.keys():
            combined_data[key] = combined_data[key][shuffle_idx]

        return combined_data

    def train_val_test_split(
        self,
        data: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Split data into train/val/test sets."""
        n_total = len(data["labels"])
        n_test = int(n_total * self.test_split)
        n_val = int((n_total - n_test) * self.val_split)

        splits = {}
        for split_name, (start, end) in [
            ("test", (0, n_test)),
            ("val", (n_test, n_test + n_val)),
            ("train", (n_test + n_val, n_total)),
        ]:
            splits[split_name] = {key: arr[start:end] for key, arr in data.items()}

        print(f"\nData splits:")
        print(
            f"  Train: {len(splits['train']['labels'])} ({len(splits['train']['labels'])/n_total*100:.1f}%)"
        )
        print(
            f"  Val:   {len(splits['val']['labels'])} ({len(splits['val']['labels'])/n_total*100:.1f}%)"
        )
        print(
            f"  Test:  {len(splits['test']['labels'])} ({len(splits['test']['labels'])/n_total*100:.1f}%)"
        )

        return splits

    def save_processed_data(
        self,
        data_splits: Dict,
        output_dir: Path,
    ):
        """Save processed data and vocabularies."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save data splits
        for split_name, split_data in data_splits.items():
            np.savez(output_dir / f"{split_name}.npz", **split_data)

        # Save vocabularies
        vocabs = {
            "drug_to_idx": self.drug_to_idx,
            "reaction_to_idx": self.reaction_to_idx,
            "idx_to_drug": self.idx_to_drug,
            "idx_to_reaction": self.idx_to_reaction,
        }

        if self.sex_to_idx:
            vocabs.update(
                {
                    "sex_to_idx": self.sex_to_idx,
                    "idx_to_sex": self.idx_to_sex,
                }
            )

        if self.route_to_idx:
            vocabs.update(
                {
                    "route_to_idx": self.route_to_idx,
                    "idx_to_route": self.idx_to_route,
                }
            )

        with open(output_dir / "vocabularies.json", "w") as f:
            json.dump(vocabs, f, indent=2)

        # Save metadata
        metadata = {
            "n_drugs": len(self.drug_to_idx),
            "n_reactions": len(self.reaction_to_idx),
            "min_drug_occurrences": self.min_drug_occurrences,
            "min_reaction_occurrences": self.min_reaction_occurrences,
            "use_age": self.use_age and "age" in data_splits["train"],
            "use_sex": self.use_sex and "sex_indices" in data_splits["train"],
            "use_route": self.use_route and "route_indices" in data_splits["train"],
            "n_sex_categories": len(self.sex_to_idx) if self.sex_to_idx else 0,
            "n_route_categories": len(self.route_to_idx) if self.route_to_idx else 0,
            "age_mean": float(self.age_mean),
            "age_std": float(self.age_std),
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nData saved to {output_dir}/")


def prepare_faers_data_with_features(
    faers_df: pd.DataFrame,
    output_dir: str = "data/processed_with_features",
    min_drug_occurrences: int = 10,
    min_reaction_occurrences: int = 10,
    negative_sampling_ratio: float = 1.0,
    use_age: bool = True,
    use_sex: bool = True,
    use_route: bool = True,
):
    """
    Prepare FAERS data with patient/drug features for deep learning.

    Args:
        faers_df: FAERS DataFrame with columns:
            - 'drugname' or 'prod_ai'
            - 'pt'
            - 'age_clean' (optional, if use_age=True)
            - 'sex' (optional, if use_sex=True)
            - 'route_clean' (optional, if use_route=True)
        output_dir: Where to save processed data
        min_drug_occurrences: Minimum drug occurrences
        min_reaction_occurrences: Minimum reaction occurrences
        negative_sampling_ratio: Negative/positive ratio
        use_age: Whether to use age as feature
        use_sex: Whether to use sex as feature
        use_route: Whether to use route as feature
    """
    preprocessor = FAERSDataPreprocessorWithFeatures(
        min_drug_occurrences=min_drug_occurrences,
        min_reaction_occurrences=min_reaction_occurrences,
        use_age=use_age,
        use_sex=use_sex,
        use_route=use_route,
    )

    # Load and filter
    df_filtered = preprocessor.load_and_filter_data(faers_df)

    # Create vocabularies
    preprocessor.create_vocabularies(df_filtered)

    # Create training data
    data = preprocessor.create_training_data_with_features(
        df_filtered,
        negative_sampling_ratio=negative_sampling_ratio,
    )

    # Split data
    data_splits = preprocessor.train_val_test_split(data)

    # Save
    preprocessor.save_processed_data(data_splits, Path(output_dir))

    return preprocessor


if __name__ == "__main__":
    print("Example: Prepare FAERS data WITH patient features")
    print("\nimport pandas as pd")
    print("from data_prep_with_features import prepare_faers_data_with_features")
    print("\ndf = pd.read_csv('your_faers_data.csv')")
    print("\npreprocessor = prepare_faers_data_with_features(")
    print("    df,")
    print("    output_dir='data/processed_with_features',")
    print("    use_age=True,")
    print("    use_sex=True,")
    print("    use_route=True,")
    print(")")
