# evaluate.py
import torch
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from dataset import BeerDataset
from utils import load_data, prepare_tokenizer
from sklearn.metrics import classification_report

def evaluate():
    # Load data
    X_test, y_test, tokenizer = load_data(split='test')
    test_loader = DataLoader(BeerDataset(X_test, y_test, tokenizer), batch_size=16)

    # Load model
    num_labels = len(set(y_test))
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model.load_state_dict(torch.load('bert_finetuned.pth'))
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()

    # Predictions
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()
