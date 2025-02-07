import pandas as pd
import numpy as np
import torch
import re
import string
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
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

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
    # Improved text cleaning
    if pd.isna(text):
        return []
    
    # Convert to lowercase and remove URLs
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
    
    # Handle special characters and emojis
    text = re.sub(r'[^\w\s\[\]]', ' ', text)
    
    # Remove multiple spaces and split
    text = ' '.join(text.split())
    
    # Tokenization and stopword removal
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

class HybridDataset(Dataset):
    def __init__(self, texts, embeddings, labels=None, tokenizer=None):
        self.texts = texts
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = 128

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
            # Intentar cargar desde caché local
            self.bert = AutoModel.from_pretrained('distilbert-base-uncased', 
                                                local_files_only=True,
                                                return_dict=True)
            print("Loaded BERT from local cache")
        except Exception as e:
            print("Local cache not found, downloading BERT model (this might take a while)...")
            self.bert = AutoModel.from_pretrained('distilbert-base-uncased',
                                                return_dict=True)
            print("BERT model downloaded successfully")
        
        # Optimize memory usage
        self.bert.config.output_hidden_states = False
        self.bert.config.output_attentions = False
        
        # Freeze BERT layers
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Unfreeze last 2 layers
        for param in self.bert.transformer.layer[-2:].parameters():
            param.requires_grad = True
        
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
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, embeddings):
        # Process BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # bert_output[0] es last_hidden_state cuando torchscript=True
        bert_cls = bert_output[0][:, 0, :]  # Tomamos el token [CLS]
        bert_encoded = self.bert_encoder(bert_cls)
        
        # Process Word2Vec
        w2v_encoded = self.w2v_encoder(embeddings)
        
        # Combine features
        combined = torch.cat((bert_encoded, w2v_encoded), dim=1)
        
        # Classify
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

def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer outside of the dataset
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', 
                                                local_files_only=True,
                                                use_fast=True)
        print("Loaded tokenizer from local cache")
    except Exception as e:
        print("Local cache not found, downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', 
                                                use_fast=True)
        print("Tokenizer downloaded successfully")
    
    # Load data
    print("Loading data...")
    df_train = pd.read_csv("nlp-getting-started/train.csv")
    df_test = pd.read_csv("nlp-getting-started/test.csv")
    
    # Preprocess text
    df_train['tokens'] = df_train['text'].apply(clean_text)
    df_test['tokens'] = df_test['text'].apply(clean_text)
    
    # Train Word2Vec with larger dimensions and window
    w2v_model = Word2Vec(
        sentences=df_train['tokens'],
        vector_size=200,
        window=10,
        min_count=1,
        workers=4,
        sg=1  # Skip-gram model
    )
    
    # Create embeddings
    def text_to_vector(tokens, model):
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if not vectors:
            return np.zeros(model.vector_size)
        # Use weighted average based on IDF
        return np.mean(vectors, axis=0)
    
    df_train['embeddings'] = df_train['tokens'].apply(lambda x: text_to_vector(x, w2v_model))
    df_test['embeddings'] = df_test['tokens'].apply(lambda x: text_to_vector(x, w2v_model))
    
    # Prepare features
    X = np.vstack(df_train['embeddings'].values)
    y = df_train['target'].values
    X_test = np.vstack(df_test['embeddings'].values)
    
    # Initialize K-Fold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Lists to store predictions
    oof_predictions = np.zeros(len(df_train))
    test_predictions = np.zeros(len(df_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold + 1}/{n_splits}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create datasets with shared tokenizer
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
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Initialize model and training components
        print(f"\nInitializing model for fold {fold + 1}...")
        model = HybridModel(embedding_dim=200)
        model = model.to(device)
        
        # Move to device and handle memory
        def move_batch_to_device(batch, device):
            return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        criterion = FocalLoss(gamma=2)
        optimizer = optim.AdamW(
            [
                {'params': model.bert.parameters(), 'lr': 1e-5},
                {'params': model.w2v_encoder.parameters()},
                {'params': model.bert_encoder.parameters()},
                {'params': model.classifier.parameters()}
            ],
            lr=2e-4,
            weight_decay=0.01
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[1e-5, 2e-4, 2e-4, 2e-4],
            epochs=10,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
        
        early_stopping = EarlyStopping(patience=3, mode='max')
        best_val_f1 = 0
        
        for epoch in range(10):
            # Training
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad()
                
                outputs = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['embeddings']
                )
                
                loss = criterion(outputs.squeeze(), batch['label'])
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                if batch_idx % 20 == 0:
                    print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        batch['input_ids'],
                        batch['attention_mask'],
                        batch['embeddings']
                    )
                    preds = torch.sigmoid(outputs.squeeze())
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(batch['label'].cpu().numpy())
            
            # Calculate metrics
            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)
            
            # Find optimal threshold
            best_threshold = 0.5
            best_f1 = 0
            
            for threshold in np.arange(0.3, 0.7, 0.01):
                f1 = f1_score(val_labels, val_preds > threshold)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            print(f"Epoch {epoch+1}, Val F1: {best_f1:.4f}, Threshold: {best_threshold:.3f}")
            
            # Early stopping
            if best_f1 > best_val_f1:
                best_val_f1 = best_f1
                torch.save({
                    'model_state': model.state_dict(),
                    'threshold': best_threshold
                }, f'best_model_fold{fold}.pth')
            
            early_stopping(best_f1)
            if early_stopping.early_stop:
                break
        
        # Load best model for predictions
        checkpoint = torch.load(f'best_model_fold{fold}.pth')
        model.load_state_dict(checkpoint['model_state'])
        threshold = checkpoint['threshold']
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            # OOF predictions
            val_dataset = HybridDataset(
                df_train['text'].iloc[val_idx].values,
                X_val
            )
            val_loader = DataLoader(val_dataset, batch_size=32)
            
            fold_preds = []
            for batch in val_loader:
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
                X_test
            )
            test_loader = DataLoader(test_dataset, batch_size=32)
            
            fold_test_preds = []
            for batch in test_loader:
                outputs = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['embeddings']
                )
                preds = torch.sigmoid(outputs.squeeze())
                fold_test_preds.extend(preds.cpu().numpy())
            
            test_predictions += (np.array(fold_test_preds) > threshold).astype(int)
    
    # Average test predictions across folds
    test_predictions = (test_predictions / n_splits > 0.5).astype(int)
    
    # Calculate OOF score
    oof_f1 = f1_score(df_train['target'], oof_predictions)
    print(f"OOF F1 Score: {oof_f1:.4f}")
    
    # Create submission
    submission = pd.DataFrame({
        'id': df_test['id'],
        'target': test_predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    train_model()