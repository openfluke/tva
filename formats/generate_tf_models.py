
import os
import numpy as np
import json

# Try importing tensorflow, handle if missing (since we know it is missing in this env)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    # tensorflowjs is optional due to install issues on Py3.13
    try:
        import tensorflowjs as tfjs
        HAS_TFJS = True
    except ImportError:
        HAS_TFJS = False
except ImportError:
    print("TensorFlow not installed. Please install it to run this script.")
    print("pip install tensorflow")
    exit(1)

# Create output directories
os.makedirs("tva/formats/output/tflite", exist_ok=True)
os.makedirs("tva/formats/output/tfjs", exist_ok=True)
os.makedirs("tva/formats/output/data", exist_ok=True)

class SwiGLU(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(SwiGLU, self).__init__(**kwargs)
        self.d_model = d_model
        self.w_gate = layers.Dense(d_model, use_bias=True)
        self.w_up = layers.Dense(d_model, use_bias=True)
        self.w_down = layers.Dense(d_model, use_bias=True)

    def call(self, x):
        # gate = SiLU(gate_proj(x)) * up_proj(x)
        gate = tf.nn.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(gate)
    
    def get_config(self):
        config = super(SwiGLU, self).get_config()
        config.update({"d_model": self.d_model})
        return config

class RMSNorm(layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        super(RMSNorm, self).__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale", 
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True
        )
        super(RMSNorm, self).build(input_shape)

    def call(self, x):
        # x * rsqrt(mean(x^2) + eps)
        mean_sq = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        norm = x * tf.math.rsqrt(mean_sq + self.eps)
        return norm * self.scale
    
    def get_config(self):
        config = super(RMSNorm, self).get_config()
        config.update({"eps": self.eps})
        return config

def create_sequence_model(vocab_size, d_model, n_head):
    inputs = layers.Input(shape=(None,), dtype="int32", name="input")
    
    # Embedding
    x = layers.Embedding(vocab_size, d_model)(inputs)
    
    # RNN
    x = layers.SimpleRNN(d_model, return_sequences=True)(x)
    
    # LSTM
    x = layers.LSTM(d_model, return_sequences=True)(x)
    
    # Attention (Self-Attention)
    # Keras MHA expects (query, value, key)
    attn_out = layers.MultiHeadAttention(num_heads=n_head, key_dim=d_model)(x, x)
    
    # LayerNorm
    x = layers.LayerNorm()(attn_out + x) # Residual
    
    # RMSNorm
    x = RMSNorm()(x)
    
    # SwiGLU
    x = SwiGLU(d_model)(x)
    
    # Output
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="sequence_model")
    return model

def create_vision_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name="input")
    
    # Conv2D
    x = layers.Conv2D(16, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    
    # Flatten & Dense
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes)(x) # Linear output
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="vision_model")
    return model

def create_audio_model(input_shape, out_channels):
    # Input shape: [Time, Channels] (Channels last in TF usually, but for consistency we use [Time, Chan])
    inputs = layers.Input(shape=input_shape, name="input")
    
    # Conv1D
    x = layers.Conv1D(16, kernel_size=3, padding="same", activation="relu")(inputs)
    outputs = layers.Conv1D(out_channels, kernel_size=3, padding="same")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="audio_model")
    return model

def export_model(model, inputs, name):
    print(f"Exporting {name}...")
    
    # Run forward pass (just to build/verify)
    output = model(inputs)
    
    # Save Data (Transposing if needed to match Torch layout if desired, 
    # but here we keep TF layout and let user handle differences)
    # TF Input: usually (Batch, [Seq/H], [W], Channels)
    np.save(f"tva/formats/output/data/tf_{name}_input.npy", inputs.numpy())
    np.save(f"tva/formats/output/data/tf_{name}_output.npy", output.numpy())

    # Export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable TF ops if needed for some layers, but standard ones should be fine
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS 
    ]
    tflite_model = converter.convert()
    with open(f"tva/formats/output/tflite/{name}.tflite", "wb") as f:
        f.write(tflite_model)
    print(f"  Saved TFLite to tva/formats/output/tflite/{name}.tflite")

    # Export TFJS
    if HAS_TFJS:
        try:
            tfjs.converters.save_keras_model(model, f"tva/formats/output/tfjs/{name}")
            print(f"  Saved TFJS to tva/formats/output/tfjs/{name}/")
        except Exception as e:
            print(f"  Failed to save TFJS (error): {e}")
    else:
        print(f"  Skipped TFJS export (tensorflowjs not installed).")

if __name__ == "__main__":
    tf.random.set_seed(42)
    
    # 1. Sequence Model
    print("--- Sequence Model ---")
    seq_model = create_sequence_model(vocab_size=100, d_model=32, n_head=4)
    # Input: [Batch=10, Seq=20]
    seq_input = tf.random.uniform((10, 20), minval=0, maxval=100, dtype=tf.int32)
    export_model(seq_model, seq_input, "sequence_model")

    # 2. Vision Model
    print("\n--- Vision Model ---")
    # TF Conv2D: [Batch, H, W, C]
    vis_model = create_vision_model((32, 32, 3), num_classes=10)
    vis_input = tf.random.normal((10, 32, 32, 3))
    export_model(vis_model, vis_input, "vision_model")

    # 3. Audio Model
    print("\n--- Audio Model ---")
    # TF Conv1D: [Batch, Time, Channels]
    aud_model = create_audio_model((100, 2), out_channels=4)
    aud_input = tf.random.normal((10, 100, 2))
    export_model(aud_model, aud_input, "audio_model")
