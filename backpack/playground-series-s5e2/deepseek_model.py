import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from category_encoders import TargetEncoder
import matplotlib.pyplot as plt

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
y = df["Price"].values  # Direct price without log transformation
X_predict = df_y

# Target Encoding for categorical variables
encoder = TargetEncoder(cols=categorical_cols)
X_train_cat = encoder.fit_transform(X[categorical_cols], y)
X_predict_cat = encoder.transform(X_predict[categorical_cols])

# Handle numerical variables
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
val_dataset = BagDataset(X_val, y_val)

# Weighted Random Sampler
sample_weights = torch.randn(len(train_dataset)).abs() + 1
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Enhanced Neural Network model
class EnhancedNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(EnhancedNeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.network(x).squeeze()

# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=40, verbose=False, delta=0.00001, path='best_model.pth'):
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
model = EnhancedNeuralNetwork(input_size).to(device)
criterion = nn.MSELoss()  # Changed to Mean Squared Error
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
early_stopping = EarlyStopping(patience=40, delta=0.00001)

# Lists to store losses for plotting
train_losses = []
val_losses = []

# Training Loop
num_epochs = 300

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    scheduler.step()
    early_stopping(avg_val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

# Plot losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

# Prediction
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

predict_dataset = BagDataset(X_predict_processed)
predict_loader = DataLoader(predict_dataset, batch_size=128, shuffle=False)

predictions = []
with torch.no_grad():
    for X_batch in predict_loader:
        X_batch = X_batch.to(device)
        y_pred = model(X_batch)
        predictions.extend(y_pred.cpu().numpy())

# Clip negative predictions
predictions = np.maximum(predictions, 0)

# Save predictions
sub = pd.DataFrame()
sub['id'] = test_ids
sub['Price'] = predictions
sub.to_csv("improved_backpack_price_submissions.csv", index=False)

print("Predictions saved to improved_backpack_price_submissions.csv")