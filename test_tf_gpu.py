import tensorflow as tf

# Check if TensorFlow detects GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {gpus}")
    try:
        # Set TensorFlow to use only the first GPU (if multiple are available)
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU is available and TensorFlow is configured to use it.")
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")
else:
    print("No GPU detected. TensorFlow is running on CPU.")

# Test TensorFlow computation on GPU if available
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print(f"Matrix multiplication result:\n{c.numpy()}")

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")
