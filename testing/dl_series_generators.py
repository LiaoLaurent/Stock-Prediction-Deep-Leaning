from tensorflow.keras.utils import Sequence
import numpy as np

class TimeSeriesDeepLOBGenerator(Sequence):
    def __init__(self, data, target, features, look_back, batch_size=32, **kwargs):
  
        self.data = data[features].values  
        self.targets = data[target].values.astype(int) 
        self.look_back = look_back
        self.batch_size = batch_size
        
        self.indices = np.arange(len(data) - look_back)
        
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        X_batch = np.array([self.data[i : i + self.look_back] for i in batch_indices])
        y_batch = np.array([self.targets[i + self.look_back] for i in batch_indices])
        
        return X_batch, y_batch
