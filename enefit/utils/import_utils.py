import os
import json

from typing import Dict, Any

def import_config(config_path: str='config/config.json') -> Dict[str, Any]:
    with open(config_path) as config_file:
        config = json.load(config_file)

    return config

def import_credential(config_path: str='config/credential.json') -> Dict[str, Any]:
    with open(config_path) as config_file:
        config = json.load(config_file)

    return config

def set_key_env() -> None:
    """
    Set credential as os environ

    """
    credential = import_credential()

    for key, value in credential.items():
        os.environ[key] = value