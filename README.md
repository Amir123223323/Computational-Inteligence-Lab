# Titanic Survival Prediction — Perceptron from Scratch

Assignment 1 of the Computational Intelligence Lab course. A single-layer
neural network (perceptron-style sigmoid neuron) is implemented **from
scratch with NumPy** — no ML library is used for training — to predict
which passengers survived the Titanic disaster. scikit-learn is used only
for the train/test split, metrics, and the confusion-matrix plot.

## Dataset

`src/titanic.csv` — the classic Kaggle Titanic dataset (891 passengers).

| Feature | Description                              | Used |
|---------|------------------------------------------|------|
| Pclass  | Ticket class (1, 2, 3)                   | ✅   |
| Sex     | `female` / `male` (encoded to 0 / 1)     | ✅   |
| SibSp   | # of siblings / spouses aboard           | ✅   |
| Parch   | # of parents / children aboard           | ✅   |
| Fare    | Ticket price                             | ✅   |
| Age     | Passenger age — **dropped** (177 missing values, no imputation) | ❌ |
| Survived| Target label (0 = not survived, 1 = survived) | target |

## Method

### Preprocessing (`pre_processing.py`)
- `Sex` encoded: `female → 0`, `male → 1`
- Every feature divided by its maximum value (max-normalization)

### Model (`perceptron_algorithm.py`)
A single neuron with a sigmoid activation, trained by **stochastic gradient
descent** (per-sample updates):

```
z     = w·x + b
ŷ     = σ(z)                       σ(z) = 1 / (1 + e⁻ᶻ)
e     = ŷ − y
δ     = e · Ź(1 − Ź)               (gradient of ½e² through the sigmoid)
w     ← w − η · δ · x
b     ← b − η · δ
```

The epoch loss is the mean of `½e²` over all training samples (MSE-style
loss). The sigmoid is clipped to `±500` to avoid overflow in `np.exp`.

### Prediction (`prediction.py`)
`ŷ ≥ 0.5 → 1` (survived), otherwise `0`.

### Hyperparameters (defaults of `train_model`)
| Parameter      | Value |
|----------------|-------|
| epochs         | 1000  |
| learning rate  | 0.001 |
| random seed    | 42    |
| test size      | 20 %  |
| decision threshold | 0.5 |

## Project Structure

```
Assignment_1/
├── README.md
├── report.docx                    # assignment report
└── src/
    ├── titanic.csv                # dataset (891 passengers)
    ├── main.py                    # entry point: train + evaluate + plots
    ├── load_data_set.py           # loads CSV, builds feature matrix X and target y
    ├── pre_processing.py          # sex encoding + max-normalization
    ├── perceptron_algorithm.py    # sigmoid neuron trained with SGD (from scratch)
    ├── activation_function.py     # numerically-safe sigmoid
    ├── prediction.py              # thresholded prediction on test data
    ├── model.py                   # pipeline: preprocess → split → train → predict
    └── plot_metrics.py            # weights bar chart, confusion matrix, loss curve
```

## Requirements

- Python 3.13 (any ≥ 3.10 should work)
- numpy, pandas, scikit-learn, matplotlib

```bash
pip install numpy pandas scikit-learn matplotlib
```

## How to Run

**In PyCharm:** simply run `src/main.py` (the project is already configured
this way — PyCharm adds both the project root and `src/` to the module
search path).

**From the terminal** (from the project root — `src/` must be on
`PYTHONPATH` because the imports mix `from src.x import ...` and
`from x import ...` styles):

```bash
# Linux / macOS / Git Bash
PYTHONPATH=src python -m src.main

# Windows PowerShell
$env:PYTHONPATH="src"; python -m src.main

# Windows cmd
set PYTHONPATH=src && python -m src.main
```

The program prints the accuracy, confusion matrix, learned weights and a
classification report, then shows three figures: model weights, confusion
matrix, and the training-loss curve.

## Sample Output

Run with the default hyperparameters (seed = 42):

```
Accuracy: 0.7821

Confusion matrix:      Classification report:
[[88 17]               Not Survived  0.80  0.84  0.82  105
 [22 52]]              Survived      0.75  0.70  0.73  74
                       accuracy                      0.78  179

Weights [Pclass, Sex, SibSp, Parch, Fare] =
    [-1.526, -2.244, -0.320, 0.120, 0.920]
```

The learned weights are interpretable: `Sex` has the largest negative
weight (encoded `female = 0`, so it raises survival probability) and `Fare`
a positive one — higher-class, more expensive tickets → higher survival
chance; `Pclass` negative because `3`rd class is the largest value.

## Known Limitations

- `Age` is excluded because of 177 missing values (imputation could
  improve the model).
- Normalization is computed on the full dataset before the train/test
  split (mild data leakage; the test maximum leaks into training).
- The train/test split is not stratified, so class balance may vary.
