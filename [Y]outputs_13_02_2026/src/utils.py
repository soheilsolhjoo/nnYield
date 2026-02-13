"""
General Utility Module for nnYield.

This module provides common helper functions for file I/O, configuration 
management, and other non-core logic operations used across the project.
"""

import yaml
import os
import logging

def load_config(path):
    """
    Loads a YAML configuration file into a Python dictionary.

    Args:
        path (str): The filesystem path to the config.yaml file.

    Returns:
        dict: The parsed configuration data.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        yaml.YAMLError: If the file is not a valid YAML document.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at: {path}")
        
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise

def save_config(config, save_dir):
    """
    Saves a configuration object (dict or dataclass) to a YAML file.

    This is primarily used to archive the exact configuration used for a 
    specific training run inside the output directory.

    Args:
        config (dict or dataclass): The configuration data to save. 
            If a dataclass is provided, it should be convertible to a dict.
        save_dir (str): The directory where config.yaml will be created.

    Returns:
        str: The full path to the saved configuration file.
    """
    os.makedirs(save_dir, exist_ok=True)
    target_path = os.path.join(save_dir, "config.yaml")
    
    # Handle dataclasses (like our Config object) if passed directly
    if hasattr(config, 'to_dict'):
        data_to_save = config.to_dict()
    elif hasattr(config, '__dict__'):
        data_to_save = vars(config)
    else:
        data_to_save = config

    with open(target_path, 'w') as f:
        yaml.dump(data_to_save, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Configuration archived to: {target_path}")
    return target_path
