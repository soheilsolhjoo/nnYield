import tensorflow as tf
import tf2onnx
import os
import yaml
import numpy as np
from .model import HomogeneousYieldModel

class Exporter:
    def __init__(self, config):
        self.config = config
        self.model = HomogeneousYieldModel(config)
        
        # Load weights
        exp_dir = os.path.join(config['training']['save_dir'], config['experiment_name'])
        weights_path = os.path.join(exp_dir, "best_model.weights.h5")
        
        if not os.path.exists(weights_path):
            weights_path = os.path.join(exp_dir, "model.weights.h5")
        
        # Build model with dummy input before loading weights
        dummy = tf.constant(np.random.randn(1, 3).astype(np.float32))
        self.model(dummy)
        print(f"Loading weights from: {weights_path}") # Debug print
        self.model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")

    def export_onnx(self):
        output_path = os.path.join(
            self.config['training']['save_dir'], 
            self.config['experiment_name'], 
            "model.onnx"
        )

        @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32)])
        def inference_func(inputs):
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(inputs)
                with tf.GradientTape(persistent=True) as tape1:
                    tape1.watch(inputs)
                    val = self.model(inputs)
                grad = tape1.gradient(val, inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)

            # Hessians
            d1 = tape2.gradient(grad[:, 0], inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            d2 = tape2.gradient(grad[:, 1], inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            d3 = tape2.gradient(grad[:, 2], inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            hess_flat = tf.concat([d1, d2, d3], axis=1)
            
            return val, grad, hess_flat

        tf2onnx.convert.from_function(
            function=inference_func,
            input_signature=[tf.TensorSpec([None, 3], tf.float32)],
            output_path=output_path,
            opset=13
        )
        print(f"Exported ONNX model to {output_path}")