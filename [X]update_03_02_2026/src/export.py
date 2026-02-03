import tensorflow as tf
import tf2onnx
import os
import yaml
import numpy as np
from .model import HomogeneousYieldModel

class Exporter:
    """
    Model Export Utility.
    
    This class handles the conversion of the trained Keras model into a standardized 
    ONNX format. Ideally suited for deployment in environments like C++, Fortran (via wrappers), 
    or Abaqus/Ansys.
    
    Key Feature:
    It exports not just the Yield Stress prediction, but also the First Derivatives (Flow Rule)
    and Second Derivatives (Tangent Modulus) as a single computational graph. 
    This is essential for implicit Finite Element solvers (Newton-Raphson).
    """
    def __init__(self, config):
        """
        Initialize the Exporter by rebuilding the model and loading trained weights.
        
        Args:
            config (dict): The full configuration dictionary.
        """
        self.config = config
        
        # 1. Rebuild the exact model architecture from config
        self.model = HomogeneousYieldModel(config)
        
        # 2. Construct Path to Weights
        weights_path = os.path.join(
            config['training']['save_dir'], 
            config['experiment_name'], 
            "model.weights.h5"
        )
        
        # 3. Build/Compile the Model
        # We must run a "dummy" forward pass to initialize the shapes of the layers
        # before we can safely load the weights.
        dummy = tf.constant(np.random.randn(1, 3).astype(np.float32))
        self.model(dummy)
        
        print(f"Loading weights from: {weights_path}")
        self.model.load_weights(weights_path)
        print(f"Successfully loaded weights.")

    def export_onnx(self):
        """
        Converts the TensorFlow model to ONNX format.
        
        The exported function signature will be:
        Inputs: 
            stress_tensor (N, 3)
            
        Outputs:
            1. val: Equivalent Stress (Scalar)
            2. grad: First Derivative d(Se)/d(sigma) (Flow Vector)
            3. hess: Second Derivative d^2(Se)/d(sigma^2) (Tangent Stiffness)
        """
        # Define output filename
        output_path = os.path.join(
            self.config['training']['save_dir'], 
            self.config['experiment_name'], 
            "model.onnx"
        )

        # --- DEFINE COMPUTATIONAL GRAPH FOR EXPORT ---
        # We wrap this in a tf.function to freeze the logic.
        # This function calculates value, gradient, and hessian simultaneously.
        @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32)])
        def inference_func(inputs):
            # Outer Tape: Records operations for Second Derivative (Hessian)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(inputs)
                
                # Inner Tape: Records operations for First Derivative (Gradient)
                with tf.GradientTape(persistent=True) as tape1:
                    tape1.watch(inputs)
                    # Forward Pass: Predict Equivalent Stress
                    val = self.model(inputs)
                
                # Calculate First Derivative (Gradient)
                # Physical Meaning: The plastic flow direction (normal to yield surface).
                grad = tape1.gradient(val, inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)

            # Calculate Second Derivative (Hessian)
            # Physical Meaning: The curvature of the yield surface (needed for stability/Jacobian).
            # We differentiate each component of the gradient vector w.r.t the input vector.
            d1 = tape2.gradient(grad[:, 0], inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO) # d(Grad_11)/d(Sigma)
            d2 = tape2.gradient(grad[:, 1], inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO) # d(Grad_22)/d(Sigma)
            d3 = tape2.gradient(grad[:, 2], inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO) # d(Grad_12)/d(Sigma)
            
            # Concatenate rows to form the flattened Hessian matrix [N, 9] or [N, 3, 3] depending on reshaping
            # Here we keep it as 3 vectors of size 3 (Total 9 components flattened)
            hess_flat = tf.concat([d1, d2, d3], axis=1)
            
            return val, grad, hess_flat

        # --- PERFORM CONVERSION ---
        # Use tf2onnx to translate the TensorFlow graph to the universal ONNX standard.
        tf2onnx.convert.from_function(
            function=inference_func,
            input_signature=[tf.TensorSpec([None, 3], tf.float32)],
            output_path=output_path,
            opset=13  # Opset 13 is a stable version supported by most runtimes
        )
        print(f"Exported ONNX model to {output_path}")