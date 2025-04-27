import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load and tokenize the data
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading dataset"):
                text = json.loads(line)['text']
                encodings = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                self.examples.append({
                    'input_ids': encodings['input_ids'].squeeze(),
                    'attention_mask': encodings['attention_mask'].squeeze()
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def create_datasets(train_file, val_file, tokenizer, max_length=2048):
    train_dataset = TextDataset(train_file, tokenizer, max_length)
    val_dataset = TextDataset(val_file, tokenizer, max_length)
    return train_dataset, val_dataset 