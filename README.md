# Drug-Adverse Event Prediction Using Deep Learning

## Project Overview

A deep learning system that predicts drug-adverse event associations from FDA FAERS (Adverse Event Reporting System) data and visualizes learned drug relationships.

---

## Quick Start

### Option 1: Use Pre-trained Model (Recommended)

With (`best_model.pt`), you can start making predictions immediately:

```bash
# 1. Install dependencies
pip install torch pandas numpy scikit-learn matplotlib scipy

# 2. Use the model for predictions
python -c "
from compare_drugs import DrugComparator

comparator = DrugComparator('outputs/my_model/best_model.pt')
risk = comparator.predict('WARFARIN', 'bleeding', age=70, sex='M', route='ORAL')
print(f'Bleeding risk: {risk:.1%}')
"

# 3. Visualize embeddings
python export_embeddings.py
python visualize_embeddings.py --method tsne
# Check outputs/visualizations/ for plots
```

### Option 2: Train from Scratch

```bash
# 1. Install dependencies
pip install torch pandas numpy scikit-learn matplotlib scipy

# 2. Download FAERS data
# Visit: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
# Download Drug, Reactions, Demographics tables for 2023 Q4 - 2025 Q1

# 3. Process raw data
# Open data_creation.ipynb in Jupyter
jupyter notebook data_creation.ipynb
# Run all cells to create faers_data.csv

# 4. Prepare training data
python data_prep_with_features.py

# 5. Train the model
python run_my_faers_data.py
# This will create: outputs/my_model/best_model.pt

# 6. Visualize results
python export_embeddings.py
python visualize_embeddings.py --method tsne
```

---

## Dataset

**Source**: FDA FAERS Database (6 quarters: 2023 Q4 through 2025 Q1)

**Raw Data**:
- Downloaded from: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
- Tables: Drug, Reactions, Outcomes, and Demographics
- Time period: 2023 Q4, 2024 Q1, 2024 Q2, 2024 Q3, 2024 Q4, 2025 Q1 (6 quarters total)
- **2,170,929 unique patient reports**
- **Thousands of unique drugs and reactions** in raw data

**Filtered Data (Model Vocabulary)**:
- **2,512 drugs** (filtered: min 10 occurrences per drug across all reports)
- **8,346 adverse reactions** (filtered: min 10 occurrences per reaction)
- **Note**: Rare drugs/reactions were excluded to reduce noise and focus on well-documented associations
- **Patient features**: Age (mean: 53.6 years, std: 20.7), Sex (M/F/Unknown), Administration route
- **~300,000+ training examples** (with 1:1 positive:negative sampling)

---

## Model Architecture

**DrugReactionModelWithFeatures** - Multi-layer Perceptron with learned embeddings

### Components:
1. **Drug Embedding Layer**: 2,512 drugs → 64 dimensions
2. **Reaction Embedding Layer**: 8,346 reactions → 64 dimensions
3. **Feature Embeddings**:
   - Sex: 3 categories → 8 dimensions
   - Route: 20 categories → 16 dimensions
   - Age: Normalized continuous value (1 dimension)
4. **MLP Classifier**:
   - Input: 153 dimensions (64 + 64 + 8 + 16 + 1)
   - Hidden layers: 153 → 128 → 64
   - Output: 1 neuron with sigmoid activation
   - Dropout: 0.3 (applied after each hidden layer)

### Training Configuration:
- **Loss**: Binary Cross-Entropy (BCELoss)
- **Optimizer**: SGD with momentum (lr=0.01, momentum=0.9, weight_decay=1e-5)
- **Batch Size**: 3,000
- **Epochs**: Up to 30 (with early stopping patience=5)
- **Learning Rate Scheduler**: ReduceLROnPlateau (reduces LR when validation AUC plateaus)
- **Train/Val/Test Split**: 70% / 10% / 20%
- **Negative Sampling**: 1:1 ratio (equal positive and randomly sampled negative examples)

### Why Training Was Fast:
- **Easy negative examples**: Randomly sampled drug-reaction pairs are mostly medically implausible (e.g., "ASPIRIN + unicorn_syndrome")
- **Clear decision boundary**: Model quickly learns to distinguish real FAERS-reported associations from random noise
- **Training time**: Converged in ~10-15 epochs, total time ~10-20 minutes on CPU
- **Trade-off**: High performance metrics, but the task is easier than real-world drug safety prediction where negatives are "plausible but unreported" associations

---

## Model Performance

| Metric | Score |
|--------|-------|
| **ROC AUC** | 99.39% |
| **Accuracy** | 96.34% |
| **Average Precision** | 99.37% |
| **Validation Loss** | 0.0976 |

**Important Context**:
- Performance measured on validation set with **random negative sampling**
- Negative examples: randomly paired drugs/reactions (most are medically implausible)
- High performance reflects the model's ability to distinguish real FAERS-reported associations from random noise
- Model operates on a **closed vocabulary** (2,512 drugs, 8,346 reactions) - cannot predict for drugs/reactions outside this set
- Real-world performance on novel or rare drug-reaction pairs would likely be lower

---

## Key Innovation: Learned Embeddings

The model learns **64-dimensional vector representations** for each drug and reaction through training.

### What the Embeddings Capture:
- **Similar adverse event profiles** → similar embeddings
- **Patient population overlap** → clustered together
- **Co-prescription patterns** → nearby in embedding space
- **Real-world clinical context** → NOT chemical structure

### Example:
- **IBUPROFEN ≈ ACETAMINOPHEN** (similarity: 0.81)
  - Both common pain relievers
  - Similar patient populations
  - Overlapping adverse event patterns

- **ATORVASTATIN ≈ METFORMIN** (similarity: 0.86)
  - Frequently co-prescribed (diabetes + cardiovascular disease)
  - Same demographic groups
  - Similar GI-related adverse events

---

## Visualization: t-SNE Embeddings

### Drug Embeddings (2D Projection)

**What it shows**: 64-dimensional drug embeddings reduced to 2D using t-SNE

**Key Patterns**:
- **Dense center**: Common outpatient medications (statins, diabetes drugs, pain relievers)
- **Periphery**: Specialized/rare drugs (biologics, orphan drugs, blood factors)
- **Proximity = Similarity**: Drugs close together have similar adverse event profiles

### Interpretation:

| Region | Drug Type | Examples |
|--------|-----------|----------|
| **Center Cluster** | Common medications, high co-prescription | Metformin, Aspirin, Ibuprofen, Statins |
| **Periphery** | Specialized/rare biologics | Vedolizumab (IBD), Idursulfase (Hunter syndrome), Factor VIII (Hemophilia) |

**Why this matters**: The model learned clinically meaningful relationships without being explicitly told about drug classes or therapeutic uses.

---

## How the Neural Network Learned This

### Training Process:

1. **Starts with random embeddings** (no structure)
2. **Receives labeled examples**:
   - (WARFARIN, bleeding, age=70, sex=M) → TRUE
   - (WARFARIN, hair_loss, age=70, sex=M) → FALSE
3. **Backpropagation adjusts embeddings** to improve predictions
4. **Result**: Drugs causing similar reactions get similar embeddings

### File Descriptions:

| File | Purpose | 
|------|---------|
| `data_creation.ipynb` | Process raw FAERS → faers_data.csv | 
| `data_prep_with_features.py` | Create vocabularies & splits |
| `run_my_faers_data.py` | Train the model |
| `compare_drugs.py` | Make predictions |
| `export_embeddings.py` | Extract embeddings |
| `visualize_embeddings.py` | Create t-SNE plots |
| `outputs/my_model/best_model.pt` | Trained model |

---

## Workflow Diagrams

### Training Pipeline (From Scratch)

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Download Raw Data                                       │
│ https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html │
│ → Download 6 quarters (2023 Q4 - 2025 Q1)                       │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Process Raw Data                                        │
│ jupyter notebook data_creation.ipynb                            │
│ → Creates: faers_data.csv (2.17M patient reports)               │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Prepare Training Data                                   │
│ python data_prep_with_features.py                               │
│ → Creates: vocabularies, train/val/test splits                  │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Train Model                                             │
│ python run_my_faers_data.py                                     │
│ → Creates: outputs/my_model/best_model.pt                       │
│ → Performance: 99.39% ROC AUC                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
```

### Inference Pipeline (Using Pre-trained Model)

```
┌─────────────────────────────────────────────────────────────────┐
│ You have: outputs/my_model/best_model.pt                        │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
              ┌───────────┴───────────┐
              ↓                       ↓
┌─────────────────────────┐  ┌──────────────────────────┐
│ Option A: Predictions   │  │ Option B: Visualizations │
│                         │  │                          │
│ python -c "             │  │ python export_embeddings │
│ from compare_drugs...   │  │ ↓                        │
│ comparator.predict()    │  │ python visualize_embed.. │
│ "                       │  │ ↓                        │
│                         │  │ See outputs/visualiz../  │
└─────────────────────────┘  └──────────────────────────┘
```

---

## Usage Examples

### 1. Predict Drug-Reaction Association
```python
from compare_drugs import DrugComparator

comparator = DrugComparator("outputs/my_model/best_model.pt")

# Single prediction
risk = comparator.predict("WARFARIN", "bleeding", age=70, sex="M", route="ORAL")
print(f"Risk probability: {risk:.3f}")
```

### 2. Compare Multiple Drugs
```python
# Compare bleeding risk across drugs
results = comparator.compare_drugs(
    ["WARFARIN", "ASPIRIN", "IBUPROFEN"],
    "bleeding",
    age=70, sex="M", route="ORAL"
)
```

### 3. Find Top Reactions for a Drug
```python
# Get top 10 most likely adverse reactions
top_reactions = comparator.top_reactions_for_drug("WARFARIN", top_k=10)
```

### 4. Visualize Embeddings
```bash
# Export embeddings
python export_embeddings.py

# Generate t-SNE plots
python visualize_embeddings.py --method tsne --perplexity 30 --annotate 20

# Generate UMAP plots (faster, preserves global structure)
python visualize_embeddings.py --method umap
```

---

## Technical Requirements

```
Python 3.8+
torch
pandas
numpy
scikit-learn
matplotlib
scipy
```

Optional:
```
umap-learn  # For UMAP visualization
```

---

## Future Directions

1. **Temporal Analysis**: Incorporate time-series patterns in adverse event reporting
2. **Explainability**: Identify which embedding dimensions correspond to specific safety patterns
3. **Multi-task Learning**: Simultaneously predict severity, outcomes, and hospitalization
4. **Negative Samples**: Improve negative sampling
5. **Interactive Dashboard**: Web interface for drug safety exploration
