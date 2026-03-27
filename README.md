# Cultural Sarcasm Detector (CSD)

A multi-task deep learning model that detects sarcasm in text and 
classifies the cultural domain it belongs to.

## Model
- **Backbone**: DistilBERT (distilbert-base-uncased)
- **Tasks**: Sarcasm detection (score 0–1) + Domain classification (7 categories)
- **Training**: Mixed precision FP16 on NVIDIA RTX 3050

## Datasets
- [Sarcasm Headlines Dataset v2](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) — Rishabh Misra (Kaggle)
- [Reddit Sarcasm Corpus](https://www.kaggle.com/datasets/danofer/sarcasm) — danofer (Kaggle)

Download both datasets and place them in `data/raw/` before running.

## Setup
```bash
conda create -n csd python=3.10 -y
conda activate csd
conda install pytorch==2.1.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install transformers==4.37.0 datasets==2.16.1 scikit-learn==1.4.0 pandas==2.1.4 numpy==1.26.3 tqdm==4.66.1 matplotlib==3.8.2 seaborn==0.13.1 accelerate==0.26.1
```

## Usage

**Train:**
```bash
python src/train.py
```

**Evaluate:**
```bash
python src/evaluate.py
```

**Interactive demo:**
```bash
python src/inference.py
```

## Results
| Metric | Score |
|--------|-------|
| Accuracy | — |
| F1 Score | — |
| AUC | — |

*(Fill in after training)*

## Project Structure
```
csd/
├── config.py          # All hyperparameters and settings
├── src/
│   ├── data_loader.py # Dataset loading and preprocessing
│   ├── preprocess.py  # Tokenization and DataLoader creation
│   ├── model.py       # DistilBERT + dual classification heads
│   ├── train.py       # Training loop with FP16 and scheduler
│   ├── evaluate.py    # Metrics, plots, and evaluation
│   └── inference.py   # Inference and interactive demo
├── data/raw/          # Place datasets here (not tracked by git)
├── models/            # Saved model weights (not tracked by git)
└── logs/              # Training plots and history
```
## Pretrained Model

The trained model weights (`models/csd_model.pt`) are stored via Git LFS.
They are downloaded automatically when you clone the repo:

git clone https://github.com/YOUR_USERNAME/csd.git

No retraining needed — run inference immediately after cloning:

python src/inference.py

## Domain Classes
`politics` · `entertainment` · `technology` · `sports` · `lifestyle` · `humor` · `general`

## Known Limitations
- Domain classification defaults toward `general` due to class imbalance 
  in Reddit training data (74% of subreddits mapped to general).
  Sarcasm detection is unaffected — trained on a balanced 50/50 split.