import pytest
import tensorflow as tf

@pytest.mark.gpu
def test_gpu_availability():
    """
    Test the availability of GPUs for TensorFlow.
    """
    # List all physical devices of type GPU
    gpu_devices = tf.config.list_physical_devices('GPU')
    # Assert that at least one GPU is available
    assert len(gpu_devices) > 0, "No GPUs available"
    # Log the number of GPUs available
    print(f"Num GPUs Available: {len(gpu_devices)}")
    # Log the GPU devices
    print("GPU Devices: ", gpu_devices)

if __name__ == "__main__":
    test_gpu_availability()
