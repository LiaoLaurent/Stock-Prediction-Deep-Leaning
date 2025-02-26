import os
import tensorflow as tf
import onnx

# Directory containing ONNX models
onnx_models_dir = "data/models/classification/"

# Convert specific ONNX model to Keras format
model_name = "classification_model_20250226_175201_epoch_10"
onnx_path = os.path.join(onnx_models_dir, f"{model_name}.onnx")
keras_path = os.path.join(onnx_models_dir, f"{model_name}.keras")

# Load ONNX model and convert to Keras
tf_model = tf.keras.layers.TFSMLayer(onnx_path, call_endpoint="serving_default")
tf_model.save(keras_path)