import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import matplotlib.pyplot as plt  # Importar matplotlib para graficar

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Load data
df = pd.read_csv("backpack/playground-series-s5e2/train.csv")
df = df.drop("id", axis=1)
df_y = pd.read_csv("backpack/playground-series-s5e2/test.csv")
test_ids = df_y["id"]
df_y = df_y.drop("id", axis=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Handle missing values in categorical variables
categorical_cols = ["Brand", "Material", "Size", "Style", "Color"]
for col in categorical_cols:
    df[col] = df[col].replace("", "Missing").fillna("Missing")
    df_y[col] = df_y[col].replace("", "Missing").fillna("Missing")

# Convert binary variables to 0/1
binary_cols = ["Laptop Compartment", "Waterproof"]
df[binary_cols] = df[binary_cols].replace({"Yes": 1, "No": 0})
df_y[binary_cols] = df_y[binary_cols].replace({"Yes": 1, "No": 0})

# Separate features and target
X = df.drop("Price", axis=1)
y = df["Price"].values  # No se aplica log1p
X_predict = df_y

# One-Hot Encoding for categorical variables
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_cat = encoder.fit_transform(X[categorical_cols])
X_predict_cat = encoder.transform(X_predict[categorical_cols])

# Handle missing values in numerical variables
numerical_cols = ["Compartments", "Weight Capacity (kg)"]
for col in numerical_cols:
    X[col] = X[col].fillna(X[col].median())
    X_predict[col] = X_predict[col].fillna(X[col].median())

# Normalize numerical variables
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X[numerical_cols])
X_predict_num = scaler.transform(X_predict[numerical_cols])

# Combine processed features
binary_train = X[binary_cols].values
binary_predict = X_predict[binary_cols].values

# Ensure no NaNs before stacking
X_train_cat = np.nan_to_num(X_train_cat, nan=0)
X_train_num = np.nan_to_num(X_train_num, nan=0)
binary_train = np.nan_to_num(binary_train, nan=0)

X_predict_cat = np.nan_to_num(X_predict_cat, nan=0)
X_predict_num = np.nan_to_num(X_predict_num, nan=0)
binary_predict = np.nan_to_num(binary_predict, nan=0)

# Stack features
X_train_processed = np.hstack([X_train_cat, X_train_num, binary_train])
X_predict_processed = np.hstack([X_predict_cat, X_predict_num, binary_predict])

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_processed,
    y,
    test_size=0.2,
    random_state=42
)

# Dataset class
class BagDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

# Create datasets and dataloaders with batch size adjusted
train_dataset = BagDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = BagDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Improved Neural Network model
class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(ImprovedNeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze()

# Early Stopping class with adjusted patience and delta
class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0.001, path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Training Configuration
input_size = X_train_processed.shape[1]
model = ImprovedNeuralNetwork(input_size)
criterion = nn.MSELoss()  # Usar MSE, pero calcular RMSE durante el entrenamiento
#optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001) # Usando Adam
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0005) # Usando AdamW
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)
early_stopping = EarlyStopping(patience=25, verbose=True, delta=0.001, path='best_model.pth')

# Lists para almacenar las pérdidas para graficar
train_losses = []
val_losses = []

# Training Loop
num_epochs = 250

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = torch.sqrt(criterion(y_pred, y_batch))  # RMSE
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss) # Guardar la pérdida de entrenamiento

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = torch.sqrt(criterion(y_pred, y_batch))  # RMSE
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss) # Guardar la pérdida de validación

    print(f"Epoch {epoch+1}, Train RMSE: {avg_train_loss:.4f}, Val RMSE: {avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)
    early_stopping(avg_val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

# Graficar las pérdidas
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss (RMSE)')
plt.plot(val_losses, label='Validation Loss (RMSE)')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

# Prediction
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

predict_dataset = BagDataset(X_predict_processed)
predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False)

predictions = []
with torch.no_grad():
    for X_batch in predict_loader:
        y_pred = model(X_batch)
        predictions.extend(y_pred.numpy())

# Save predictions
sub = pd.DataFrame()
sub['id'] = test_ids
sub['Price'] = predictions
sub.to_csv("improved_claude_submissions.csv", index=False)

print("Predictions saved to improved_claude_submissions.csv")
