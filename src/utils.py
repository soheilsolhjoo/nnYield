import yaml
import os

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    target_path = os.path.join(save_dir, "config.yaml")
    with open(target_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to: {target_path}")