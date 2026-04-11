import argparse
import json

from arguments.config import config as cfg
from modules.gui import GUI

def deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override, only replacing leaves."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            deep_update(base[key], val)
        else:
            base[key] = val
    return base

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="")
    
    parser.add_argument("--outputs-path", type=str, default="")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--vae-name", type=str, default="")
    
    args = parser.parse_args()
    
    # User config for GUI and training
    config = cfg  # start from defaults
    if args.config_path:
        with open(args.config_path) as fp:
            user_config = json.load(fp)
        deep_update(config, user_config)
    
    # Overide with use inputs
    if args.outputs_path != "": cfg["training"]["outputs"]["path"]=args.outputs_path
    if args.model_name != "": cfg["model"]["main-name"]=args.model_name
    if args.vae_name != "": cfg["model"]["vae-name"]=args.vae_name
    if args.dataset_path != "": cfg["training"]["dataset"]["path"]=args.dataset_path
    
        
    gui = GUI(cfg)
    gui.run()