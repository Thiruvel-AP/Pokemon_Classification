# Pokémon Multi-Target Classification: Random Forest + GridSearchCV

A **three-model multi-target classification system** built on 809 Pokémon records, implementing an input-routing architecture that automatically selects the correct prediction model based on which attributes are provided. Each model is optimised independently via **5-fold stratified GridSearchCV** across 96 hyperparameter combinations with three custom scoring functions, then persisted via joblib for inference.

> Built with: `scikit-learn` · `pandas` · `NumPy` · `matplotlib` · `seaborn` · `joblib` · Python

---

## What This Demonstrates

- **Multi-target classification**: three separate supervised models each predicting a different target variable from overlapping but distinct feature subsets
- **Input-routing inference function**: single `predict_pokemon()` entry point dispatches to the correct model based on which arguments are present, not hard-coded conditionals
- **Production-ready pipeline design**: `StandardScaler → RandomForestClassifier` in a scikit-learn `Pipeline`, preventing data leakage between CV folds
- **Class imbalance handling**: `class_weight="balanced"` on all three models
- **Custom multi-metric GridSearch**: simultaneous optimisation on F1, Hamming loss, and Precision; best model selected by minimum Hamming loss (`refit="HAMMING_LOSS"`)

---

## Dataset

| Property | Detail |
|---|---|
| Source | `pokemon.csv` (local, included) |
| Observations | 809 Pokémon |
| Columns | `Name`, `Type1`, `Type2`, `Evolution` |
| `Type1` unique values | 18 types (Bug · Dark · Dragon · Electric · Fairy · Fighting · Fire · Flying · Ghost · Grass · Ground · Ice · Normal · Poison · Psychic · Rock · Steel · Water) |
| `Type2` coverage | 405 / 809 have a secondary type (50.1%): 404 NaN |
| `Evolution` coverage | 32 / 809 have a recorded next evolution (4.0%): 777 NaN |

**NaN handling:** All `NaN` values replaced with `"0"` via `fillna(0)`. The string `"0"` serves as the sentinel for "none": present after `LabelEncoder` and detected at output via `if output != '0'` guard in the inference function.

---

## Three-Model Architecture

```
Input: (Name, Type1, Type2) ──► Model 1 ──► Predict: Evolution
Input: (Name, Evolution)    ──► Model 2 ──► Predict: Type1 (Primary)
Input: (Name, Type1, Evolution) ► Model 3 ──► Predict: Type2 (Secondary)

                    Routing logic (predict_pokemon()):
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    name+type1      name+evolution  name+type1
    (no evolution)  (no type1)      +evolution
          │              │              │
     model_evolution  model_type1  model_type2
         .pkl            .pkl          .pkl
```

**Router implementation**: `predict_pokemon()` checks argument presence via `if name and type1 and evolution is None` / `elif name and evolution and not type1` / `elif name and type1 and evolution`: mutually exclusive cases covering all three prediction scenarios.

---

## Pipeline Architecture

```python
model_pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('classifier', RandomForestClassifier(
        random_state=42,
        class_weight="balanced"   # handles class imbalance (Evolution: 4% coverage)
    ))
])
```

**Why `StandardScaler` before Random Forest?** RF is scale-invariant, but `StandardScaler` is included in the pipeline to ensure the design pattern is robust if the classifier is swapped for a scale-sensitive model (e.g. SVM, LogisticRegression) without changing the pipeline structure.

---

## Hyperparameter Optimisation

### Search Space

| Hyperparameter | Values searched |
|---|---|
| `classifier__max_depth` | 3, 5, 10, 20, 30, 50 |
| `classifier__min_samples_split` | 2, 5, 10, 20 |
| `classifier__n_estimators` | 300, 500, 700, 1000 |

**Total candidates:** 6 × 4 × 4 = **96 combinations**  
**CV folds:** 5 (`KFold(n_splits=5, shuffle=True, random_state=42)`)  
**Total fits per model:** 96 × 5 = **480 fits**  
**Parallelism:** `n_jobs=-1`: all CPU cores used simultaneously

### Custom Scoring Functions

```python
def custom_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted", zero_division=1)

def custom_hamming(y_true, y_pred):
    return -hamming_loss(y_true, y_pred)    # negated so higher = better

def custom_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average="weighted", zero_division=1)
```

`refit="HAMMING_LOSS"`: the model with minimum Hamming loss (maximum of negated value) is selected as the best estimator. Hamming loss is used as the primary selection criterion because it penalises per-label errors proportionally: appropriate for multi-class problems with class imbalance.

---

## Best Parameters & CV Results (from actual training output)

| Model | Target | `max_depth` | `min_samples_split` | `n_estimators` | Best CV Hamming Loss |
|---|---|---|---|---|---|
| Model 1 | Evolution | 20 | 2 | 300 | **0.0569** |
| Model 2 | Type1 | 20 | 2 | 300 | 0.784 |
| Model 3 | Type2 | 20 | 2 | 1000 | 0.560 |

**Model 1 (Evolution) achieves Hamming loss of 0.057**: high performance explained by the dataset's class distribution: 96% of Pokémon have no evolution (label = `"0"`), making this a highly imbalanced problem where the `class_weight="balanced"` correction matters. Models 2 and 3 face harder multi-class problems across 18 type categories.

---

## Label Encoding

Four separate `LabelEncoder` instances: one per column: each fitted independently and saved as `.pkl` for inference-time inverse transformation:

```python
name_encoder      → name_encoder.pkl       # 809 unique Pokémon names → integers
type1_encoder     → type1_encoder.pkl      # 18 primary types → integers
type2_encoder     → type2_encoder.pkl      # 18 types + "0" sentinel → integers
evolution_encoder → evolution_encoder.pkl  # evolution names + "0" sentinel → integers
```

Inverse transform at inference: `evolution_encoder.inverse_transform([prediction])[0]` recovers the original Pokémon name or `"0"` for "no evolution", displayed as `"Nothing....!!!"`.

---

## Inference Function (`predict_pokemon`)

```python
predict_pokemon(name=None, type1=None, type2=None, evolution=None)
```

**Three routing scenarios:**

| Scenario | Required args | Missing arg | Model used | Output |
|---|---|---|---|---|
| 1 | `name`, `type1` | `evolution=None` | `model_evolution.pkl` | Evolution name |
| 2 | `name`, `evolution` | `type1=None` | `model_type1.pkl` | Primary type |
| 3 | `name`, `type1`, `evolution` |: | `model_type2.pkl` | Secondary type |

**Demo prediction (actual output):**
```python
predict_pokemon(name="marshadow", type1="Fighting", evolution="0")
# Predicting Type2 (Secondary Type)...
# → 'Predicted Output: Ghost'
```
Marshadow is indeed Fighting/Ghost type: prediction is correct.

---

## Saved Artefacts

```
model_evolution.pkl    # Best Pipeline: StandardScaler + RF (max_depth=20, n_estimators=300)
model_type1.pkl        # Best Pipeline: StandardScaler + RF (max_depth=20, n_estimators=300)
model_type2.pkl        # Best Pipeline: StandardScaler + RF (max_depth=20, n_estimators=1000)
name_encoder.pkl       # LabelEncoder for Pokémon names
type1_encoder.pkl      # LabelEncoder for primary types
type2_encoder.pkl      # LabelEncoder for secondary types
evolution_encoder.pkl  # LabelEncoder for evolution names
```

All artefacts saved with `joblib.dump()` and loaded at inference via `joblib.load()`: no retraining required for prediction.

---

## Project Structure

```
Pokemon_Classification/
├── pokemon_classification.ipynb   # Full pipeline: EDA → encoding → training → inference
├── pokemon.csv                    # Dataset: 809 rows × 4 columns
├── model_evolution.pkl            # Saved RF pipeline (Evolution predictor)
├── model_type1.pkl                # Saved RF pipeline (Type1 predictor)
├── model_type2.pkl                # Saved RF pipeline (Type2 predictor)
├── name_encoder.pkl
├── type1_encoder.pkl
├── type2_encoder.pkl
└── evolution_encoder.pkl
```

---

## Quickstart

```bash
git clone https://github.com/Thiruvel-AP/Pokemon_Classification
cd Pokemon_Classification
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

Open and run `pokemon_classification.ipynb` in Jupyter: cells run top-to-bottom: EDA → encoding → GridSearchCV training (runs 3 × 480 fits, ~5–10 min on CPU) → inference demo.

**Predict without retraining** (models already saved):

```python
import joblib as jb
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load encoders and model
name_enc = jb.load("name_encoder.pkl")
type1_enc = jb.load("type1_encoder.pkl")
model    = jb.load("model_type2.pkl")

# Predict secondary type for Charizard (Fire, with evolution=None/0)
predict_pokemon(name="charizard", type1="Fire", evolution="0")
```

---

## Sector Applications

| Sector | Transferable methodology |
|---|---|
| **Technology / ML Engineering** | Multi-target classification routing pattern; `GridSearchCV` with custom multi-metric scorers; `Pipeline` design preventing data leakage; joblib model serialisation for production inference |
| **Healthcare & Omics Research** | Multi-label classification on biological entities (gene → pathway, protein → function, compound → target type); Hamming loss as selection criterion applies directly to multi-label genomic annotation tasks |
| **Finance & Analytics** | Multi-output prediction with missing/sparse labels (e.g. asset class → sector → sub-sector when sector is unknown); class imbalance handling via `class_weight="balanced"` applies to fraud detection and rare event classification |

---

## Author

**Thiruvel Andagurunathan Pandian**: MSc Data Science, University of Bristol  
Designing reproducible ML pipelines from EDA through training, evaluation, and production-ready serialisation.  
📍 Bristol, UK · **Eligible for Skilled Worker Visa sponsorship** · Open to UK roles

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/thiruvel-a-p)
[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/Thiruvel-AP)
