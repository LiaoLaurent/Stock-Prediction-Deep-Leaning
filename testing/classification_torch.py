import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json
import signal
import sys
import logging
from pathlib import Path

import dotenv

from sklearn.preprocessing import StandardScaler, MinMaxScaler

dotenv.load_dotenv()

start_date = "2024-10-01"  # os.getenv("START_DATE")  # Début janvier
end_date = "2024-12-16"    # Fin mars

print(start_date)
print(end_date)

stock_name = os.getenv("STOCK_NAME")

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

# Configure paths
MODEL_DIR = "data/models/classification"
METRICS_DIR = "data/metrics"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"training_{current_time}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Global variables for interruption handling
stop_training = False
current_model = None
current_metrics = None

def signal_handler(signum, frame):
    global stop_training
    logging.info("Signal d'interruption reçu. Arrêt propre en cours...")
    stop_training = True
    save_model_and_metrics()

def save_model_and_metrics():
    global current_model, current_metrics
    if current_model is not None:
        try:
            # Save model
            model_name = f"classification_model_{current_time}_interrupted"
            model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
            torch.save(current_model.state_dict(), model_path)
            logging.info(f"Modèle sauvegardé: {model_path}")

            # Save metrics
            if current_metrics is not None:
                metrics_path = os.path.join(METRICS_DIR, f"{model_name}_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(current_metrics, f, indent=4)
                logging.info(f"Métriques sauvegardées: {metrics_path}")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde: {str(e)}")

class ModelCheckpoint:
    def __init__(self):
        self.best_val_loss = float('inf')
        
    def __call__(self, model, epoch, logs):
        global current_model, current_metrics
        current_model = model
        current_metrics = logs

        # Save model
        model_name = f"classification_model_{current_time}_epoch_{epoch+1}"
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
        torch.save(model.state_dict(), model_path)
        
        # Save metrics
        metrics_path = os.path.join(METRICS_DIR, f"{model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(logs, f, indent=4)
        
        logging.info(f"Epoch {epoch+1} terminée - Modèle et métriques sauvegardés")

# Install signal handler
signal.signal(signal.SIGINT, signal_handler)

print(f"PyTorch version: {torch.__version__}")
print(f"GPU disponible: {torch.cuda.is_available()}")

# Feature lists
standard_features = ["DMP_5", "DMP_10", "DMN_5", "DMN_10", "MACD_8_21_5", "AO_5_10",
    "EMA_15", "MA_20", "KAMA_3_2_10", "CO", "C2O2", "C3O3",
    "net_add_ask_size", "net_add_bid_size", "Bollinger_Upper", "Bollinger_Lower"
]

minmax_features = [
    "ADX_10", "ADX_7", "ADX_5", "STOCHk_7_3_3", "STOCHd_7_3_3", "RSI_7", "time_since_open"
]

unscaled_features = ['market_session']
features = standard_features + minmax_features + unscaled_features

# Calculer les indices une seule fois
standard_indices = [features.index(f) for f in standard_features]
minmax_indices = [features.index(f) for f in minmax_features]
unscaled_indices = [features.index(f) for f in unscaled_features]

sampling_rate = "500ms"  # Échantillonnage plus fréquent
prediction_column = "Target_close"
batch_size = 128  # Augmentation de la taille du batch pour plus d'efficacité
epochs = 20
look_back = 64  # Plus de contexte temporel

# Data preprocessing
from tf_preprocessing import process_and_combine_data


# used AAPL
all_data = process_and_combine_data(start_date, end_date, data_folder="/home/janis/3A/EA/HFT_QR_RL/data/smash4/DB_MBP_10/" + stock_name, sampling_rate=sampling_rate)

print(all_data.columns)
all_data.head()

all_data.Target_close.value_counts()

# Plotting
sns.set_style("darkgrid")
plt.figure(figsize=(12, 6))
plt.plot(all_data.index, all_data["mid_price_close"], label="Mid Price Mean", color="blue")
plt.xlabel("Time")
plt.ylabel("Mid Price Close")
plt.title("Mid Price Close Over Time")
plt.legend()
plt.show()

# Data Splitting
train_size = int(len(all_data) * 0.7)
val_size = int(len(all_data) * 0.1)
test_size = len(all_data) - train_size - val_size

train_df = all_data.iloc[:train_size, :]
val_df = all_data.iloc[train_size:train_size + val_size, :]
test_df = all_data.iloc[train_size + val_size:, :]

class TimeSeriesDataset(Dataset):
    def __init__(self, data, target, look_back, oversample=False):
        self.data = data[features].values
        self.targets = data[target].values.astype(int)
        self.look_back = look_back
        
        # Pre-calculate indices
        self.indices = np.arange(len(data) - look_back)
        
        # Fit scalers on all data once
        self.standard_scaler = StandardScaler().fit(self.data[:, standard_indices])
        self.minmax_scaler = MinMaxScaler().fit(self.data[:, minmax_indices])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq_idx = self.indices[idx]
        seq = self.data[seq_idx:seq_idx + self.look_back]
        target = self.targets[seq_idx + self.look_back]
        
        # Transform using pre-fitted scalers
        seq_standard = self.standard_scaler.transform(seq[:, standard_indices])
        seq_minmax = self.minmax_scaler.transform(seq[:, minmax_indices])
        seq_unscaled = seq[:, unscaled_indices]
        
        # Concatenate all features
        seq_normalized = np.hstack((seq_standard, seq_minmax, seq_unscaled))
        return torch.FloatTensor(seq_normalized), torch.LongTensor([target])[0]

# Create datasets and dataloaders
train_dataset = TimeSeriesDataset(train_df, prediction_column, look_back)
val_dataset = TimeSeriesDataset(val_df, prediction_column, look_back)
test_dataset = TimeSeriesDataset(test_df, prediction_column, look_back)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

class SimpleClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True)
        self.fc = nn.Linear(128, 3)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Prendre le dernier timestep
        x = self.fc(x)
        return x

# Training
model = SimpleClassifier(len(features)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
checkpoint = ModelCheckpoint()

logging.info("Début de l'entraînement...")
try:
    metrics_history = []
    for epoch in range(epochs):
        if stop_training:
            break
            
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        # Calculate metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_correct / train_total,
            'val_loss': val_loss / len(val_loader),
            'val_acc': val_correct / val_total
        }
        metrics_history.append(metrics)
        
        # Checkpoint
        checkpoint(model, epoch, metrics)
        
        logging.info(f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {metrics['train_loss']:.4f} - "
                    f"Train Acc: {metrics['train_acc']:.4f} - "
                    f"Val Loss: {metrics['val_loss']:.4f} - "
                    f"Val Acc: {metrics['val_acc']:.4f}")
    
    current_model = model
    current_metrics = {'history': metrics_history}
    save_model_and_metrics()
    logging.info("Entraînement terminé avec succès")
    
except Exception as e:
    logging.error(f"Erreur pendant l'entraînement: {str(e)}")
    save_model_and_metrics()
