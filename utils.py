# utils.py
import pandas as pd
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder

def load_data(split='train'):
    df = pd.read_csv('beer_data.csv')
    tokenizer = prepare_tokenizer()
    le = load_label_encoder()

    if split == 'train':
        data = df.sample(frac=0.8, random_state=42)
    else:
        data = df.drop(df.sample(frac=0.8, random_state=42).index)

    return data['Description'], le.transform(data['label']), tokenizer

def prepare_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

def load_label_encoder():
    df = pd.read_csv('beer_data.csv')
    le = LabelEncoder()
    le.fit(df['label'])
    return le
