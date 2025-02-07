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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Inicializar lematizador y stopwords

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))  # Cambia a 'spanish' si el texto está en español

# Función para limpiar texto

def clean_text(text):
    # 1. Eliminar URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 2. Eliminar menciones y hashtags (opcional: convertir hashtags a palabras)
    text = re.sub(r'@\w+|#', '', text)
    # 3. Eliminar caracteres especiales y puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 4. Convertir a minúsculas
    text = text.lower()
    # 5. Tokenización
    tokens = word_tokenize(text)
    # 6. Eliminar stopwords y lematizar
    cleaned_tokens = [
        lemmatizer.lemmatize(token) for token in tokens
        if token not in stop_words and token.isalpha()  # Solo palabras alfabéticas
    ]

    # Unir tokens en un solo texto (opcional, dependiendo de cómo uses Word2Vec)
    cleaned_text = ' '.join(cleaned_tokens)
    cleaned_text = cleaned_text.split()
    
    return cleaned_text

# Load data

df_train = pd.read_csv("nlp-getting-started/train.csv")
df_test = pd.read_csv("nlp-getting-started/test.csv")

df_train['tokens'] = df_train['text'].apply(clean_text)
df_test['tokens'] = df_test['text'].apply(clean_text)

w2v_model = Word2Vec(sentences=df_train['tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Función para convertir un texto tokenizado a un vector de embeddings (promediando las palabras)
def text_to_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Convertir textos a embeddings
df_train['embeddings'] = df_train['tokens'].apply(lambda x: text_to_vector(x, w2v_model))
df_test['embeddings'] = df_test['tokens'].apply(lambda x: text_to_vector(x, w2v_model))

x_train = df_train.drop(["target","id","text","tokens"], axis=1)
y_train = df_train["target"].values

x_test = df_test.drop(["id","text","tokens"], axis=1)

# check x_test x_train columns



# Handle missing values in categorical variables
categorical_cols = ["keyword"]
# categorize the keyword column
label_encoder = LabelEncoder()
x_train['keyword'] = label_encoder.fit_transform(x_train['keyword'].astype(str))
x_test['keyword'] = label_encoder.transform(x_test['keyword'].astype(str))

x_train['location'] = x_train['location'].apply(lambda x: 0 if pd.isnull(x) else 1)
x_test['location'] = x_test['location'].apply(lambda x: 0 if pd.isnull(x) else 1)

# check missing values

X = np.column_stack([x_train['location'], x_train['keyword'], np.vstack(x_train['embeddings'])])
y = y_train
X_test = np.column_stack([x_test['location'], x_test['keyword'], np.vstack(x_test['embeddings'])])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Dataset
class BagDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

# Mejorada Neural Network
class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze()

# Training Configuration
input_size = X_train.shape[1]
model = ImprovedNeuralNetwork(input_size)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Create DataLoader
train_dataset = BagDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = BagDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Training
num_epochs = 50
best_loss = float('inf')
best_model_state_dict = None

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = torch.sqrt(criterion(y_pred, y_batch))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = torch.sqrt(criterion(y_pred, y_batch))
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)
    
    print(f"Epoch {epoch+1}, Train RMSE: {avg_train_loss:.4f}, Val RMSE: {avg_val_loss:.4f}")
    
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_model_state_dict = model.state_dict()
        torch.save(best_model_state_dict, 'best_model.pth')







