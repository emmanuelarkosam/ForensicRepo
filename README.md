
# Forensic Analysis Reproducibility Repository

This repository accompanies the manuscript "Investigating Methods for Forensic Analysis of Social Media Data to Support Criminal Investigations."

All data is synthetic and matches the original study's schema. Follow scripts to reproduce:
```bash
pip install -r model/requirements.txt
python scripts/generate_synthetic_data.py
python scripts/preprocess_data.py
python scripts/train_models.py
```
Results (metrics and figures) will be in `artefacts/`, `results.json`, and `figures/`.

Included:
- `datasets/`: synthetic data files + documentation
- `preprocessing/`: preprocessing steps documentation
- `model/`: hyperparameter configurations and dependencies
- `scripts/`: code to generate data, preprocess, and train/evaluate models
- `docs/`: failure-case analysis
