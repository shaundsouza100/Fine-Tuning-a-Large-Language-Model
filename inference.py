# inference.py
import torch
from transformers import BertForSequenceClassification
from utils import prepare_tokenizer, load_label_encoder

def predict(text):
    tokenizer = prepare_tokenizer()
    label_encoder = load_label_encoder()

    encoding = tokenizer.encode_plus(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
    model.load_state_dict(torch.load('bert_finetuned.pth'))
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1)
        label = label_encoder.inverse_transform(pred.cpu().numpy())[0]

    print(f"Predicted Label: {label}")

if __name__ == "__main__":
    sample_text = "This is a crisp, refreshing beer with citrus notes."
    predict(sample_text)
