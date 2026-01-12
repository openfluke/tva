import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tf2onnx
from safetensors.numpy import save_file as save_safetensors

# Determine absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Create output directories
os.makedirs(os.path.join(OUTPUT_DIR, "simple_models/tflite"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "simple_models/saved_model"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "simple_models/onnx"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "simple_models/safetensors"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "simple_models/data"), exist_ok=True)

def export_simple_model(model, input_data, name):
    print(f"--- Exporting {name} ---")
    
    # 1. Save Input/Output Data (Numpy)
    output_data = model(input_data)
    np.save(os.path.join(OUTPUT_DIR, f"simple_models/data/{name}_input.npy"), input_data.numpy())
    np.save(os.path.join(OUTPUT_DIR, f"simple_models/data/{name}_output.npy"), output_data.numpy())
    print("  [x] Saved .npy data")

    # 2. Export SavedModel (for TFJS later)
    saved_model_path = os.path.join(OUTPUT_DIR, f"simple_models/saved_model/{name}")
    # Use tf.saved_model.save() to create a standard SavedModel
    try:
        tf.saved_model.save(model, saved_model_path)
        print("  [x] Saved SavedModel")
    except Exception as e:
        print(f"  [!] Failed SavedModel: {e}")

    # 3. Export TFLite
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = os.path.join(OUTPUT_DIR, f"simple_models/tflite/{name}.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print("  [x] Saved TFLite")
    except Exception as e:
        print(f"  [!] Failed TFLite: {e}")

    # 4. Export ONNX
    try:
        onnx_path = os.path.join(OUTPUT_DIR, f"simple_models/onnx/{name}.onnx")
        # tf2onnx expects a concrete function or similar, but for Keras model we can often use:
        spec = (tf.TensorSpec(input_data.shape, input_data.dtype, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=onnx_path)
        print("  [x] Saved ONNX")
    except Exception as e:
        print(f"  [!] Failed ONNX: {e}")

    # 5. Export Safetensors
    # We extract weights from the Keras model and save them named by their weight name
    try:
        tensors_dict = {}
        for weight in model.weights:
            # Clean up name: 'dense/kernel:0' -> 'dense.kernel'
            clean_name = weight.name.replace(":0", "").replace("/", ".")
            tensors_dict[clean_name] = weight.numpy()
        
        st_path = os.path.join(OUTPUT_DIR, f"simple_models/safetensors/{name}.safetensors")
        save_safetensors(tensors_dict, st_path)
        print("  [x] Saved Safetensors")
    except Exception as e:
        print(f"  [!] Failed Safetensors: {e}")


def create_dense_model():
    inputs = keras.Input(shape=(10,), name="input")
    x = layers.Dense(16, activation="relu", name="dense1")(inputs)
    outputs = layers.Dense(2, activation="softmax", name="output")(x)
    return keras.Model(inputs, outputs, name="simple_dense")

def create_conv2d_model():
    inputs = keras.Input(shape=(28, 28, 1), name="input")
    x = layers.Conv2D(4, kernel_size=3, padding="same", activation="relu", name="conv1")(inputs)
    outputs = layers.Flatten()(x)
    return keras.Model(inputs, outputs, name="simple_conv2d")

def create_rnn_model():
    inputs = keras.Input(shape=(5, 10), name="input") # [Batch, Time, Feat]
    # SimpleRNN
    outputs = layers.SimpleRNN(8, return_sequences=False, name="rnn1")(inputs)
    return keras.Model(inputs, outputs, name="simple_rnn")

def create_lstm_model():
    inputs = keras.Input(shape=(5, 10), name="input")
    outputs = layers.LSTM(8, return_sequences=False, name="lstm1")(inputs)
    return keras.Model(inputs, outputs, name="simple_lstm")

def create_embedding_model():
    # Vocab 100, embed dim 16
    inputs = keras.Input(shape=(10,), dtype="int32", name="input")
    outputs = layers.Embedding(100, 16, name="embedding1")(inputs)
    return keras.Model(inputs, outputs, name="simple_embedding")

if __name__ == "__main__":
    tf.random.set_seed(42)
    
    # Dense
    m_dense = create_dense_model()
    x_dense = tf.random.normal((1, 10))
    export_simple_model(m_dense, x_dense, "dense")

    # Conv2D
    m_conv = create_conv2d_model()
    x_conv = tf.random.normal((1, 28, 28, 1))
    export_simple_model(m_conv, x_conv, "conv2d")

    # RNN
    m_rnn = create_rnn_model()
    x_rnn = tf.random.normal((1, 5, 10))
    export_simple_model(m_rnn, x_rnn, "rnn")

    # LSTM
    m_lstm = create_lstm_model()
    x_lstm = tf.random.normal((1, 5, 10))
    export_simple_model(m_lstm, x_lstm, "lstm")

    # Embedding
    m_embed = create_embedding_model()
    x_embed = tf.random.uniform((1, 10), minval=0, maxval=99, dtype=tf.int32)
    export_simple_model(m_embed, x_embed, "embedding")

    print("\nAll simple models generated.")
