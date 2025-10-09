"""
Handler para manejo de configuración con tipado estricto.
"""
from typing import Any, Dict, Optional, Union
import json
import yaml
from pathlib import Path


class ConfigHandler:
    """Handler para manejo de configuración."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> 'ConfigHandler':
        """Carga configuración desde un diccionario."""
        self.config = config_dict.copy()
        return self
    
    def load_from_file(self, file_path: Union[str, Path]) -> 'ConfigHandler':
        """Carga configuración desde un archivo (JSON o YAML)."""
        path = Path(file_path)
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        self.config = config_dict
        return self
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Guarda configuración a un archivo (JSON o YAML)."""
        path = Path(file_path)
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de configuración por clave."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> 'ConfigHandler':
        """Establece un valor de configuración."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        return self
    
    def update(self, updates: Dict[str, Any]) -> 'ConfigHandler':
        """Actualiza la configuración con nuevos valores."""
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        
        deep_update(self.config, updates)
        return self
    
    def validate_required(self, required_keys: list[str]) -> None:
        """Valida que todas las claves requeridas estén presentes."""
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Obtiene una sección completa de configuración."""
        return self.get(section, {})