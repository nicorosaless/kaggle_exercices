import pandas as pd
import numpy as np
import torch
import re
import string
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import BatchSampler, SequentialSampler

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'max' and score <= self.best_score + self.min_delta) or \
             (self.mode == 'min' and score >= self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def clean_text(text):
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '[USER]', text)
    text = re.sub(r'#\w+', '[HASHTAG]', text)
    text = re.sub(r'\d+', '[NUM]', text)
    text = re.sub(r'[^\w\s\[\]]', ' ', text)
    text = ' '.join(text.split())
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

class HybridDataset(Dataset):
    def __init__(self, texts, embeddings, labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'embeddings': self.embeddings[idx]
        }
        
        if self.labels is not None:
            item['label'] = self.labels[idx]
            
        return item

class HybridModel(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        print("Loading BERT model...")
        try:
            self.bert = AutoModel.from_pretrained('microsoft/deberta-v3-base', 
                                                local_files_only=True)
            print("Loaded DeBERTa from local cache")
        except Exception:
            print("Local cache not found, downloading DeBERTa model...")
            self.bert = AutoModel.from_pretrained('microsoft/deberta-v3-base')
            print("DeBERTa model downloaded successfully")
        
        self.bert.gradient_checkpointing_enable()
        
        self.w2v_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim - (embedding_dim % 8),
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.w2v_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.bert_encoder = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, embeddings):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        bert_cls = bert_output.last_hidden_state[:, 0, :]
        bert_encoded = self.bert_encoder(bert_cls)
        
        embeddings = embeddings.unsqueeze(1)
        w2v_attended, _ = self.w2v_attention(embeddings, embeddings, embeddings)
        w2v_encoded = self.w2v_encoder(w2v_attended.squeeze(1))
        
        combined = torch.cat((bert_encoded, w2v_encoded), dim=1)
        logits = self.classifier(combined)
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    scaler = GradScaler()
    
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', 
                                                local_files_only=True,
                                                use_fast=True)
        print("Loaded tokenizer from local cache")
    except Exception as e:
        print("Local cache not found, downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', 
                                                use_fast=True)
        print("Tokenizer downloaded successfully")
    
    print("Loading data...")
    df_train = pd.read_csv("nlp-getting-started/train.csv")
    df_test = pd.read_csv("nlp-getting-started/test.csv")
    
    df_train['tokens'] = df_train['text'].apply(clean_text)
    df_test['tokens'] = df_test['text'].apply(clean_text)
    
    w2v_model = Word2Vec(
        sentences=df_train['tokens'],
        vector_size=296,
        window=10,
        min_count=1,
        workers=4,
        sg=1,
        epochs=20
    )
    
    def text_to_vector(tokens, model):
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if not vectors:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)
    
    df_train['embeddings'] = df_train['tokens'].apply(lambda x: text_to_vector(x, w2v_model))
    df_test['embeddings'] = df_test['tokens'].apply(lambda x: text_to_vector(x, w2v_model))
    
    X = np.vstack(df_train['embeddings'].values)
    y = df_train['target'].values
    X_test = np.vstack(df_test['embeddings'].values)
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(df_train))
    test_predictions = np.zeros(len(df_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold + 1}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_dataset = HybridDataset(
            df_train['text'].iloc[train_idx].values,
            X_train,
            y_train,
            tokenizer=tokenizer
        )
        val_dataset = HybridDataset(
            df_train['text'].iloc[val_idx].values,
            X_val,
            y_val,
            tokenizer=tokenizer
        )
        
        train_batch_size = 16
        eval_batch_size = 32
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            num_workers=4,
            pin_memory=True
        )
        
        model = HybridModel(embedding_dim=296)
        model.to(device)
        
        criterion = FocalLoss(gamma=2)
        optimizer = optim.AdamW(
            [
                {'params': model.bert.parameters(), 'lr': 1e-5},
                {'params': model.w2v_encoder.parameters(), 'lr': 2e-4},
                {'params': model.bert_encoder.parameters(), 'lr': 2e-4},
                {'params': model.classifier.parameters(), 'lr': 2e-4}
            ],
            weight_decay=0.01
        )

        # Setup learning rate scheduler
        num_training_steps = len(train_loader) * 5
        num_warmup_steps = num_training_steps // 10
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        n_epochs = 5
        accumulation_steps = 4
        best_val_f1 = 0
        early_stopping = EarlyStopping(patience=2)
        
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                if batch['input_ids'].size(0) == 1:
                    continue  # Skip single-sample batches
                
                if np.random.random() > 0.5:
                    mixed_embeddings, labels_a, labels_b, lam = mixup_data(
                        batch['embeddings'],
                        batch['label']
                    )
                    batch['embeddings'] = mixed_embeddings
                    
                    with autocast('cuda'):
                        outputs = model(
                            batch['input_ids'],
                            batch['attention_mask'],
                            batch['embeddings']
                        )
                        
                        outputs = outputs.squeeze()
                        if len(outputs.shape) == 0:
                            outputs = outputs.unsqueeze(0)
                        
                        loss = lam * criterion(outputs, labels_a) + \
                               (1 - lam) * criterion(outputs, labels_b)
                else:
                    with autocast('cuda'):
                        outputs = model(
                            batch['input_ids'],
                            batch['attention_mask'],
                            batch['embeddings']
                        )
                        
                        outputs = outputs.squeeze()
                        if len(outputs.shape) == 0:
                            outputs = outputs.unsqueeze(0)
                        
                        loss = criterion(outputs, batch['label'])
                
                loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                
                if batch_idx % 20 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    
                    with autocast('cuda'):
                        outputs = model(
                            batch['input_ids'],
                            batch['attention_mask'],
                            batch['embeddings']
                        )
                        loss = criterion(outputs.squeeze(), batch['label'])
                        val_losses.append(loss.item())
                    
                    preds = torch.sigmoid(outputs.squeeze())
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(batch['label'].cpu().numpy())
            
            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)
            avg_val_loss = np.mean(val_losses)
            
            best_threshold = 0.5
            best_f1 = 0
            
            for threshold in np.arange(0.3, 0.7, 0.01):
                f1 = f1_score(val_labels, val_preds > threshold)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val F1: {best_f1:.4f}, Threshold: {best_threshold:.3f}")
            
            if best_f1 > best_val_f1:
                best_val_f1 = best_f1
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'threshold': torch.tensor(best_threshold),
                    'epoch': epoch,
                    'best_f1': best_f1
                }
                torch.save(save_dict, f'best_model_fold{fold}.pth')
            
            if early_stopping(best_f1):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model and make predictions
        checkpoint = torch.load(f'best_model_fold{fold}.pth', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        threshold = float(checkpoint['threshold'])
        
        model.eval()
        with torch.no_grad():
            # OOF predictions
            val_dataset = HybridDataset(
                df_train['text'].iloc[val_idx].values,
                X_val,
                tokenizer=tokenizer
            )
            val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, num_workers=4, pin_memory=True)
            
            fold_preds = []
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                with autocast('cuda'):
                    outputs = model(
                        batch['input_ids'],
                        batch['attention_mask'],
                        batch['embeddings']
                    )
                preds = torch.sigmoid(outputs.squeeze())
                fold_preds.extend(preds.cpu().numpy())
            
            oof_predictions[val_idx] = (np.array(fold_preds) > threshold).astype(int)
            
            # Test predictions
            test_dataset = HybridDataset(
                df_test['text'].values,
                X_test,
                tokenizer=tokenizer
            )
            test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, num_workers=4, pin_memory=True)
            
            fold_test_preds = []
            for batch in test_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                with autocast('cuda'):
                    outputs = model(
                        batch['input_ids'],
                        batch['attention_mask'],
                        batch['embeddings']
                    )
                preds = torch.sigmoid(outputs.squeeze())
                fold_test_preds.extend(preds.cpu().numpy())
            
            test_predictions += (np.array(fold_test_preds) > threshold).astype(int)
    
    test_predictions = (test_predictions / n_splits > 0.5).astype(int)
    
    oof_f1 = f1_score(df_train['target'], oof_predictions)
    print(f"\nFinal OOF F1 Score: {oof_f1:.4f}")
    
    submission = pd.DataFrame({
        'id': df_test['id'],
        'target': test_predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    pd.DataFrame({
        'id': df_train['id'],
        'true_target': df_train['target'],
        'pred_target': oof_predictions
    }).to_csv('oof_predictions.csv', index=False)
    print("OOF predictions saved to oof_predictions.csv")

if __name__ == "__main__":
    train_model()