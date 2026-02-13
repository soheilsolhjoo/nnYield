"""
Model Export and Deployment Module for nnYield.

This module handles the conversion of trained PINN models into industry-standard 
formats like ONNX. It includes a specialized wrapper that embeds mathematical 
gradients and Hessians directly into the exported graph.
"""

import tensorflow as tf
import tf2onnx
import os
import numpy as np
from .model import HomogeneousYieldModel
from .config import Config

class Exporter:
    """
    Handles model serialization and cross-platform deployment.
    """
    def __init__(self, config: Config):
        """
        Initializes the exporter and restores the best trained weights.
        """
        self.config = config
        self.model = HomogeneousYieldModel(config)
        
        # 1. Locate trained weights
        exp_dir = os.path.join(config.training.save_dir, config.experiment_name)
        weights_path = os.path.join(exp_dir, "best_model.weights.h5")
        
        if not os.path.exists(weights_path):
            weights_path = os.path.join(exp_dir, "model.weights.h5")
        
        # 2. Reconstruct and Load
        # Standard build call to initialize layer shapes
        dummy_input = tf.zeros((1, 3))
        _ = self.model(dummy_input)
        
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print(f"📦 Exporter: Loaded weights from {weights_path}")
        else:
            print(f"⚠️ Warning: No weights found at {weights_path}. Exporting uninitialized model.")

    def export_onnx(self, opset=13):
        """
        TRICK: THE GRADIENT-GRAPH WRAPPER.
        
        Most neural network exports only provide the output value. For material 
        science (FEA), we also need the normality (Gradient) and the tangent 
        stiffness (Hessian). 
        
        This method wraps the model in a custom TensorFlow function that uses 
        AutoDiff to calculate derivatives. When converted to ONNX, these 
        mathematical operations become part of the static graph. This means 
        FEA software can get the Stress, Strain-Ratio, and Stiffness in a 
        single call without needing its own AutoDiff engine.
        """
        output_path = os.path.join(
            self.config.training.save_dir, 
            self.config.experiment_name, 
            "model.onnx"
        )

        @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32)])
        def inference_graph(inputs):
            """
            Custom graph that returns [Value, Gradient, Flattened_Hessian].
            """
            with tf.GradientTape(persistent=True) as tape_outer:
                tape_outer.watch(inputs)
                with tf.GradientTape() as tape_inner:
                    tape_inner.watch(inputs)
                    val = self.model(inputs)
                
                # Normality Vector (dF/dS)
                grad = tape_inner.gradient(val, inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)

            # Tangent Stiffness Components (Hessian)
            # We calculate column by column to avoid high-memory batch jacobians in the ONNX graph
            d1 = tape_outer.gradient(grad[:, 0], inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            d2 = tape_outer.gradient(grad[:, 1], inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            d3 = tape_outer.gradient(grad[:, 2], inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            
            # Pack Hessian into a (N, 9) flat tensor for the ONNX output
            hessian_flat = tf.concat([d1, d2, d3], axis=1)
            
            return val, grad, hessian_flat

        # Convert to ONNX
        print(f"🚀 Converting to ONNX (Opset {opset})...")
        tf2onnx.convert.from_function(
            function=inference_graph,
            input_signature=[tf.TensorSpec([None, 3], tf.float32)],
            output_path=output_path,
            opset=opset
        )
        print(f"✅ Model successfully exported to: {output_path}")
