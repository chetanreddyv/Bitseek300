import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import BitSeek
import wandb
from tqdm import tqdm
import math

def train(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    weight_decay,
    warmup_steps,
    device,
    save_path
):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler with warmup
    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * (step / warmup_steps)
        return learning_rate * 0.1 * (1 + math.cos(math.pi * (step - warmup_steps) / (num_epochs * len(train_loader) - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Calculate loss
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Log to wandb
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': scheduler.get_last_lr()[0]
            })
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                logits = model(input_ids, attention_mask)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        wandb.log({'val_loss': val_loss})
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'New best model saved with validation loss: {val_loss:.4f}')

def main():
    # Initialize wandb
    wandb.init(project="tinylm", config={
        "learning_rate": 2e-4,
        "weight_decay": 0.1,
        "warmup_steps": 375,
        "batch_size": 16,
        "num_epochs": 3,
        "model_size": "300M"
    })
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = BitSeek(
        vocab_size=50257,  # GPT-2 vocabulary size
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=2048,
        dropout=0.1
    ).to(device)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Set special tokens
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # TODO: Implement your dataset loading logic here
    # train_dataset = YourDataset(...)
    # val_dataset = YourDataset(...)
    
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Train the model
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
        learning_rate=2e-4,
        weight_decay=0.1,
        warmup_steps=375,
        device=device,
        save_path="tinylm_best.pt"
    )
    
    wandb.finish()

if __name__ == "__main__":
    main() 