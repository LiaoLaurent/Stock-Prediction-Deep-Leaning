# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _kg_hide-input=true _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 0.857016, "end_time": "2022-04-05T06:45:25.031603", "exception": false, "start_time": "2022-04-05T06:45:24.174587", "status": "completed"}
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import os
import json
import signal
import sys
import logging
from pathlib import Path
import dotenv

dotenv.load_dotenv()

stock_name = os.getenv("STOCK_NAME")

# Configuration du GPU
tf.random.set_seed(0)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Configuration des chemins
MODEL_DIR = "data/models/classification"
METRICS_DIR = "data/metrics"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configuration du logging
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

# Variables globales pour la gestion de l'interruption
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
            # Sauvegarde du modèle en format .keras
            model_name = f"classification_model_{current_time}_interrupted"
            model_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
            current_model.save(model_path)
            logging.info(f"Modèle sauvegardé en format .keras: {model_path}")

            # Sauvegarde des métriques
            if current_metrics is not None:
                metrics_path = os.path.join(METRICS_DIR, f"{model_name}_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(current_metrics, f, indent=4)
                logging.info(f"Métriques sauvegardées: {metrics_path}")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde: {str(e)}")

class ModelCheckpointCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global current_model, current_metrics
        current_model = self.model
        current_metrics = logs

        # Sauvegarde du modèle en format .keras
        model_name = f"classification_model_{current_time}_epoch_{epoch+1}"
        model_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
        self.model.save(model_path)
        
        # Sauvegarde des métriques
        metrics_path = os.path.join(METRICS_DIR, f"{model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(logs, f, indent=4)
        
        logging.info(f"Epoch {epoch+1} terminée - Modèle et métriques sauvegardés")

        if stop_training:
            self.model.stop_training = True

# Installation du gestionnaire de signal
signal.signal(signal.SIGINT, signal_handler)

print(tf.__version__)
print("GPU disponible:", tf.config.list_physical_devices('GPU'))

# +
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
# -

sampling_rate = "500ms"  # Échantillonnage plus fréquent
prediction_column = "Target_close"
batch_size = 128  # Augmentation de la taille du batch pour plus d'efficacité
epochs = 20
look_back = 64  # Plus de contexte temporel

# +
from tf_preprocessing import process_and_combine_data

start_date = os.getenv("START_DATE")  # Début janvier
end_date = os.getenv("END_DATE")    # Fin mars

# used AAPL
all_data = process_and_combine_data(start_date, end_date, data_folder="/home/janis/3A/EA/HFT_QR_RL/data/smash4/DB_MBP_10/" + stock_name, sampling_rate=sampling_rate)

print(all_data.columns)

all_data.head()
# -

all_data.Target_close.value_counts()

# +
sns.set_style("darkgrid")

plt.figure(figsize=(12, 6))
plt.plot(all_data.index, all_data["mid_price_close"], label="Mid Price Mean", color="blue")
plt.xlabel("Time")
plt.ylabel("Mid Price Close")
plt.title("Mid Price Close Over Time")
plt.legend()
plt.show()

# +
# Data Splitting
train_size = int(len(all_data) * 0.7)
val_size = int(len(all_data) * 0.1)
test_size = len(all_data) - train_size - val_size

train_df = all_data.iloc[:train_size, :]
val_df = all_data.iloc[train_size:train_size + val_size, :]
test_df = all_data.iloc[train_size + val_size:, :]

# +
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter

standard_indices = [features.index(f) for f in standard_features]
minmax_indices = [features.index(f) for f in minmax_features]
unscaled_indices = [features.index(f) for f in unscaled_features]

class TimeSeriesScalerGenerator(Sequence):
    def __init__(self, data, target, look_back, batch_size=128, oversample=False, **kwargs):
        super().__init__(**kwargs)
        
        # Utiliser des chunks de données pour la mémoire
        self.data = data[features].values
        self.targets = data[target].values.astype(int)
        self.look_back = look_back
        self.batch_size = batch_size
        self.indices = np.arange(len(data) - look_back)
        
        if oversample:
            self._oversample_minority_classes()
            
        # Pré-calculer les indices des features pour plus d'efficacité
        self.standard_indices = [features.index(f) for f in standard_features]
        self.minmax_indices = [features.index(f) for f in minmax_features]
        self.unscaled_indices = [features.index(f) for f in unscaled_features]
        
        # Initialiser les scalers une seule fois
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        """Generates one batch of data."""
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = np.array([self.data[i:i + self.look_back] for i in batch_indices])
        
        X_batch = np.empty((len(batch_indices), self.look_back, len(features)), dtype=np.float32)
        y_batch = self.targets[batch_indices + self.look_back]
        
        # Normalisation par batch pour plus d'efficacité
        for i, seq in enumerate(batch_data):
            X_batch[i] = self._normalize_sequence(seq)
        
        return X_batch, y_batch
    
    def _extract_true_labels(self):
        """Extract all true labels for the entire dataset."""
        return np.array([self.targets[i + self.look_back] for i in self.indices])

    def _oversample_minority_classes(self):
        """Oversample minority classes by duplicating their sequences."""
        # Count class distribution
        class_counts = Counter(self.targets)
        max_count = max(class_counts.values())

        # Collect indices for each class
        class_indices = {label: np.where(self.targets == label)[0] for label in class_counts}

        # Oversample minority classes
        oversampled_indices = []
        for label, indices in class_indices.items():
            repeat_count = max_count // len(indices)  # Number of times to repeat the minority class
            oversampled_indices.extend(np.repeat(indices, repeat_count))
            oversampled_indices.extend(np.random.choice(indices, max_count % len(indices), replace=True))

        # Update indices and targets
        self.indices = np.array(oversampled_indices)
        self.targets = self.targets[self.indices]

    def _normalize_sequence(self, seq):
        # Normalisation optimisée
        seq_standard = self.standard_scaler.fit_transform(seq[:, self.standard_indices])
        seq_minmax = self.minmax_scaler.fit_transform(seq[:, self.minmax_indices])
        seq_unscaled = seq[:, self.unscaled_indices]
        
        return np.hstack((seq_standard, seq_minmax, seq_unscaled))

# -

train_gen = TimeSeriesScalerGenerator(train_df, prediction_column, look_back, batch_size, oversample=True)
val_gen = TimeSeriesScalerGenerator(val_df, prediction_column, look_back, batch_size)
test_gen = TimeSeriesScalerGenerator(test_df, prediction_column, look_back, batch_size)

# +
from tensorflow.keras.callbacks import EarlyStopping
from keras import layers

# early_stop = EarlyStopping(
#     monitor="val_loss",  # Monitor validation loss
#     patience=3,          # Stop after 3 epochs of no improvement
#     restore_best_weights=True  # Restore best weights after stopping
# )

# Architecture du modèle - Version simple mais plus grande
def create_model(input_size):
    inputs = layers.Input(shape=(look_back, input_size))

    # First LSTM layer - plus grand
    x = layers.LSTM(512, return_sequences=True)(inputs)
    x = layers.BatchNormalization()(x)

    # MultiHeadAttention simple
    attn_output = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Second LSTM layer - plus grand
    x = layers.LSTM(256, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Dense layers - plus larges
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Train Model
with tf.device('/GPU:0'):
    logging.info("Début de l'entraînement...")
    try:
        model = create_model(len(features))
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[ModelCheckpointCallback()],
        )
        
        current_model = model
        current_metrics = history.history
        save_model_and_metrics()
        
        logging.info("Entraînement terminé avec succès")
    except Exception as e:
        logging.error(f"Erreur pendant l'entraînement: {str(e)}")
        save_model_and_metrics()
# -
