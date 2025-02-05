import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Cargar datos
df = pd.read_csv("backpack/playground-series-s5e2/train.csv")
df = df.drop("id", axis=1)
df_y = pd.read_csv("backpack/playground-series-s5e2/test.csv")
test_ids = df_y["id"]
df_y = df_y.drop("id", axis=1)

# Manejar valores faltantes en categóricas
categorical_cols = ["Brand", "Material", "Size", "Style", "Color"]
for col in categorical_cols:
    df[col] = df[col].replace("", "Missing").fillna("Missing")
    df_y[col] = df_y[col].replace("", "Missing").fillna("Missing")

# Convertir variables binarias a 0/1
binary_cols = ["Laptop Compartment", "Waterproof"]
df[binary_cols] = df[binary_cols].replace({"Yes": 1, "No": 0})
df_y[binary_cols] = df_y[binary_cols].replace({"Yes": 1, "No": 0})

# Separar features y target
X = df.drop("Price", axis=1)
y = df["Price"].values
X_predict = df_y

# One-Hot Encoding para categóricas
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
X_train_cat = encoder.fit_transform(X[categorical_cols])
X_predict_cat = encoder.transform(X_predict[categorical_cols])

# Manejar valores faltantes en numéricas
numerical_cols = ["Compartments", "Weight Capacity (kg)"]
for col in numerical_cols:
    X[col] = X[col].fillna(X[col].median())
    X_predict[col] = X_predict[col].fillna(X[col].median())

# Normalizar numéricas
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X[numerical_cols])
X_predict_num = scaler.transform(X_predict[numerical_cols])

# Combinar features procesadas
binary_train = X[binary_cols].values
binary_predict = X_predict[binary_cols].values

# Ensure no NaNs before stacking
X_train_cat = np.nan_to_num(X_train_cat, nan=0)
X_train_num = np.nan_to_num(X_train_num, nan=0)
binary_train = np.nan_to_num(binary_train, nan=0)

X_predict_cat = np.nan_to_num(X_predict_cat, nan=0)
X_predict_num = np.nan_to_num(X_predict_num, nan=0)
binary_predict = np.nan_to_num(binary_predict, nan=0)

# Stacking features
X_train_processed = np.hstack([X_train_cat, X_train_num, binary_train])
X_predict_processed = np.hstack([X_predict_cat, X_predict_num, binary_predict])

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_train_processed, y, test_size=0.2, random_state=42)

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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Create DataLoader
train_dataset = BagDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = BagDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Training
num_epochs = 250
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

# Predictions
model.load_state_dict(best_model_state_dict)
model.eval()

with torch.no_grad():
    y_pred = model(torch.tensor(X_predict_processed, dtype=torch.float32))

# Guardar predicciones
sub = pd.DataFrame()
sub['id'] = test_ids
sub['Price'] = y_pred.numpy()
sub.to_csv("submissionss.csv", index=False)

print("Predictions saved to submissions.csv SIGMA")
