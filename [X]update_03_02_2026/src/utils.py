import yaml
import os

"""
Utility Module for Configuration Management.

This module handles the loading and saving of experiment configurations (YAML files).
Centralizing this logic ensures consistent parameter handling across training, 
evaluation, and sanity checking scripts.
"""

def load_config(path):
    """
    Loads a YAML configuration file into a Python dictionary.
    
    Args:
        path (str): The file path to the .yaml config file.
        
    Returns:
        dict: The parsed configuration parameters.
        
    Raises:
        FileNotFoundError: If the specified path does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, save_dir):
    """
    Saves the current configuration dictionary to a YAML file in the output directory.
    
    This is crucial for **Experiment Reproducibility**. It ensures that even if 
    you change the original 'config.yaml' later, you always have a copy of exactly 
    what settings produced a specific trained model.
    
    Args:
        config (dict): The configuration dictionary to save.
        save_dir (str): The directory where 'config.yaml' will be written.
    """
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    target_path = os.path.join(save_dir, "config.yaml")
    
    with open(target_path, 'w') as f:
        # default_flow_style=False ensures the YAML is written in block format (readable)
        # rather than inline JSON-like format.
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"Configuration saved to: {target_path}")