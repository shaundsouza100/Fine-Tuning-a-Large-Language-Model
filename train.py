# train.py

import torch
from transformers import BertForSequenceClassification, BertTokenizer, get_scheduler
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
df = pd.read_csv('beer_data.csv')
df = df[['Description', 'Categories']].dropna()
df['Categories'] = df['Categories'].str.strip().str.lower()

# Filter categories with at least 50 samples
counts = df['Categories'].value_counts()
df = df[df['Categories'].isin(counts[counts >= 50].index)]

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['Categories'])

# Train-validation-test split
X = df['Description']
y = df['label']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Dataset class
class BeerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        encoding = self.tokenizer.encode_plus(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
num_labels = df['label'].nunique()
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# DataLoaders
train_loader = DataLoader(BeerDataset(X_train, y_train, tokenizer), batch_size=16, shuffle=True)
val_loader = DataLoader(BeerDataset(X_val, y_val, tokenizer), batch_size=16)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
num_training_steps = 3 * len(train_loader)
lr_scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Train loss = {total_loss / len(train_loader):.4f}")

# Save checkpoint
torch.save(model.state_dict(), 'checkpoints/bert_beer_epoch_3.pt')
print("Model saved!")
