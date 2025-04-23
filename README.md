# Fine-Tuning BERT for Beer Style Classification

## Project Overview

This project fine-tunes the pre-trained **BERT-base-uncased** model to classify beer styles based on textual descriptions. The model is trained to predict the beer style (e.g., IPA, Stout, Lager) from descriptive reviews using natural language processing techniques.

## Dataset

- **Source**: [Wikiliq Beer Reviews Dataset](https://www.kaggle.com/datasets/limtis/wikiliq-dataset)
- **Features**:
  - `Description`: Textual review of the beer.
  - `Categories`: Beer style labels.

For this project, a filtered subset of **1000 samples** across **40 beer styles** was used for proof-of-concept fine-tuning.

## Setup Instructions

### Clone the repository
```bash
# Clone the repository
 git clone <repo_link>
 cd <repo_directory>
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Code

### 1. Train the Model
```bash
python train.py
```

### 2. Evaluate the Model
```bash
python evaluate.py
```

### 3. Inference Example
```bash
python inference.py --text "This is a rich, malty beer with notes of chocolate and coffee."
```

## Project Structure
```
├── train.py         # Fine-tuning script
├── evaluate.py      # Model evaluation script
├── inference.py     # Inference pipeline
├── requirements.txt # Project dependencies
└── README.md        # Project overview and instructions
```

## Reproducibility Notes

- The model was trained using **Google Colab** with **GPU acceleration**.
- **Random seeds** were set for reproducibility, but minor variations in results may occur depending on hardware.
- Subsampled dataset (1000 rows) was used to ensure faster training iterations.

## References

- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. arXiv preprint arXiv:1810.04805.
- Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Kaggle: Wikiliq Beer Reviews Dataset: https://www.kaggle.com/datasets/limtis/wikiliq-dataset

Fine-Tuning a Large Language Model
