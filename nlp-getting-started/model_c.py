import pandas as pd
import numpy as np
import torch
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def clean_text(text):
    # Mantener la función de limpieza existente
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(token) for token in tokens
        if token not in stop_words and token.isalpha()
    ]
    cleaned_text = ' '.join(cleaned_tokens)
    cleaned_text = cleaned_text.split()
    return cleaned_text

class BagDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        
        # Apply self-attention
        attn_output, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = attn_output.squeeze(1)
        
        # Classify
        return self.classifier(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def train_model():
    # Cargar y preprocesar datos (mantener el código existente hasta la creación de dataloaders)
    df_train = pd.read_csv("nlp-getting-started/train.csv")
    df_test = pd.read_csv("nlp-getting-started/test.csv")

    # Preprocess text
    df_train['tokens'] = df_train['text'].apply(clean_text)
    df_test['tokens'] = df_test['text'].apply(clean_text)

    # Train Word2Vec
    w2v_model = Word2Vec(sentences=df_train['tokens'], vector_size=100, window=5, min_count=1, workers=4)

    def text_to_vector(tokens, model):
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    df_train['embeddings'] = df_train['tokens'].apply(lambda x: text_to_vector(x, w2v_model))
    df_test['embeddings'] = df_test['tokens'].apply(lambda x: text_to_vector(x, w2v_model))

    # Prepare features
    x_train = df_train.drop(["target", "id", "text", "tokens"], axis=1)
    y_train = df_train["target"].values
    x_test = df_test.drop(["id", "text", "tokens"], axis=1)

    # Handle categorical variables
    label_encoder = LabelEncoder()
    x_train['keyword'] = label_encoder.fit_transform(x_train['keyword'].astype(str))
    x_test['keyword'] = label_encoder.transform(x_test['keyword'].astype(str))

    x_train['location'] = x_train['location'].apply(lambda x: 0 if pd.isnull(x) else 1)
    x_test['location'] = x_test['location'].apply(lambda x: 0 if pd.isnull(x) else 1)

    X = np.column_stack([x_train['location'], x_train['keyword'], np.vstack(x_train['embeddings'])])
    y = y_train
    X_test = np.column_stack([x_test['location'], x_test['keyword'], np.vstack(x_test['embeddings'])])

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)
    X_test_normalized = scaler.transform(X_test)

    # Create datasets and dataloaders
    train_dataset = BagDataset(X_train_normalized, y_train)
    val_dataset = BagDataset(X_val_normalized, y_val)
    test_dataset = BagDataset(X_test_normalized)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model and training components
    input_size = X_train.shape[1]
    model = ImprovedNeuralNetwork(input_size)
    
    # Usar focal loss y optimizer mejorado
    criterion = FocalLoss(gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
    
    # Usar scheduler cíclico
    num_epochs = 100
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    early_stopping = EarlyStopping(patience=15)
    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Aplicar mixup
            lam = np.random.beta(0.5, 0.5)
            index = torch.randperm(X_batch.size(0))
            mixed_x = lam * X_batch + (1 - lam) * X_batch[index]
            
            y_pred = model(mixed_x)
            loss = lam * criterion(y_pred, y_batch.unsqueeze(1)) + \
                   (1 - lam) * criterion(y_pred, y_batch[index].unsqueeze(1))
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_preds.extend(y_pred.detach().cpu().numpy())
            train_targets.extend(y_batch.cpu().numpy())
        
        # Calcular métricas de entrenamiento
        train_preds = np.array(train_preds) > 0.5
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation phase con threshold optimization
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                val_preds.extend(y_pred.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
        
        # Encontrar el mejor threshold
        thresholds = np.arange(0.3, 0.7, 0.01)
        best_threshold = 0.5
        best_epoch_val_acc = 0
        
        for threshold in thresholds:
            val_pred_binary = np.array(val_preds) > threshold
            acc = accuracy_score(val_targets, val_pred_binary)
            if acc > best_epoch_val_acc:
                best_epoch_val_acc = acc
                best_threshold = threshold
        
        # Usar el mejor threshold
        val_pred_binary = np.array(val_preds) > best_threshold
        val_acc = accuracy_score(val_targets, val_pred_binary)
        
        print(f"Epoch {epoch+1}")
        print(f"Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        print(f"Best Threshold: {best_threshold:.3f}")
        print("--------------------")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'model_state': model.state_dict(),
                'threshold': best_threshold
            }
            torch.save(best_model_state, 'best_model.pth')
        
        # Early stopping
        early_stopping(1 - val_acc)  # Usando accuracy como métrica
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load best model for predictions
    model.load_state_dict(best_model_state['model_state'])
    best_threshold = best_model_state['threshold']
    
    # Generate predictions for test set
    model.eval()
    test_predictions = []
    
    with torch.no_grad():
        for X_batch in test_loader:
            y_pred = model(X_batch)
            predicted = (y_pred.data > best_threshold).float()
            test_predictions.extend(predicted.squeeze().numpy())
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': df_test['id'],
        'target': test_predictions
    })
    
    # Save predictions
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    train_model()