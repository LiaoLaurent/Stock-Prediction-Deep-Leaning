# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _kg_hide-input=true _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 0.857016, "end_time": "2022-04-05T06:45:25.031603", "exception": false, "start_time": "2022-04-05T06:45:24.174587", "status": "completed"}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(0)

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

sampling_rate = "1s"
prediction_column = ["returns", "returns_volatilities"]
batch_size = 16
epochs = 10
look_back = 32

# +
from tf_preprocessing import process_and_combine_data

start_date = "2024-10-02"
end_date = "2024-10-02"

all_data = process_and_combine_data(start_date, end_date, data_folder="../AAPL_data", sampling_rate=sampling_rate)

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
    def __init__(self, data, target, look_back, batch_size=32, oversample=False, **kwargs):
        """
        Custom Timeseries Generator with per-sequence scaling and optional oversampling.
        
        Args:
            data (pd.DataFrame): DataFrame with feature columns.
            target (str): Target column name.
            look_back (int): Number of past time steps per sample.
            batch_size (int): Batch size.
            oversample (bool): Whether to oversample minority classes.
        """
        super().__init__(**kwargs)  # Call the parent class constructor with kwargs

        self.data = data[features].values  # Extract feature matrix
        self.targets = data[target].values.astype(int)  # Extract target labels
        self.look_back = look_back
        self.batch_size = batch_size
        self.indices = np.arange(len(data) - look_back)
        self.true_labels = self._extract_true_labels()

        # Oversample minority classes if enabled
        if oversample:
            self._oversample_minority_classes()

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        """Generates one batch of data."""
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        # Extract sequences efficiently using list slicing
        batch_data = np.array([self.data[i : i + self.look_back] for i in batch_indices])

        # Preallocate arrays for batch
        X_batch = np.empty((len(batch_indices), self.look_back, len(features)), dtype=np.float32)
        y_batch = np.empty((len(batch_indices), self.targets.shape[1]), dtype=np.float32)

        # Scale each sequence individually
        for i, seq in enumerate(batch_data):
            standard_scaler = StandardScaler()
            minmax_scaler = MinMaxScaler(feature_range=(0, 1))

            seq_standard = standard_scaler.fit_transform(seq[:, standard_indices])
            seq_minmax = minmax_scaler.fit_transform(seq[:, minmax_indices])
            seq_unscaled = (
                seq[:, unscaled_indices] 
                if unscaled_features else np.empty((self.look_back, 0))
            )

            X_batch[i] = np.hstack((seq_standard, seq_minmax, seq_unscaled))
            y_batch[i] = self.targets[batch_indices[i] + self.look_back]

        return X_batch, y_batch
    
    def _extract_true_labels(self):
        """Extract all true labels for the entire dataset."""
        return np.array([self.targets[i + self.look_back] for i in self.indices])

    def _oversample_minority_classes(self):
        """Oversample minority classes by duplicating their sequences."""
        # Count class distribution
        class_counts = Counter(self.true_labels)
        max_count = max(class_counts.values())

        # Collect indices for each class
        class_indices = {label: np.where(self.true_labels == label)[0] for label in class_counts}

        # Oversample minority classes
        oversampled_indices = []
        for label, indices in class_indices.items():
            repeat_count = max_count // len(indices)  # Number of times to repeat the minority class
            oversampled_indices.extend(np.repeat(indices, repeat_count))
            oversampled_indices.extend(np.random.choice(indices, max_count % len(indices), replace=True))

        # Update indices and true_labels
        self.indices = np.array(oversampled_indices)
        self.true_labels = self.true_labels[self.indices]


# -

train_gen = TimeSeriesScalerGenerator(train_df, prediction_columns, look_back, batch_size, oversample=True)
val_gen = TimeSeriesScalerGenerator(val_df, prediction_columns, look_back, batch_size)
test_gen = TimeSeriesScalerGenerator(test_df, prediction_columns, look_back, batch_size)

# +
from tensorflow.keras.callbacks import EarlyStopping
from keras import layers, Model, optimizers

input_size = len(features)

inputs = layers.Input(shape=(look_back, input_size))

# First LSTM layer
x = layers.LSTM(128, return_sequences=True)(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)  # Dropout after LSTM

# Second LSTM layer
x = layers.LSTM(64, return_sequences=False)(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

# Dense layers
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.2)(x)

# Two outputs for regression
returns_output = layers.Dense(1, name="returns")(x)  # Output for returns
volatility_output = layers.Dense(1, name="returns_volatilities")(x)  # Output for volatilities

# Create model
model = Model(inputs=inputs, outputs=[returns_output, volatility_output])

optimizer = optimizers.Adam(learning_rate=0.001)

# Compile the model
model.compile(
    optimizer=optimizer, 
    loss={"returns": "mean_squared_error", "returns_volatilities": "mean_squared_error"},
    metrics={"returns": "mae", "returns_volatilities": "mae"}
)

# Train Model
model.fit(train_gen, validation_data=val_gen, epochs=epochs)

# +
# Evaluate the model
test_results = model.evaluate(test_gen)
print("Test Loss (Returns):", test_results[1])
print("Test MAE (Returns):", test_results[3])

print("Test Loss (Volatilities):", test_results[2])
print("Test MAE (Volatilities):", test_results[4])
