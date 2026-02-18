"""
Configuration Management

Handles loading and managing configuration settings for the data lake.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class DataLakeConfig:
    """Data lake configuration"""
    base_path: str = "./data"
    storage_backend: str = "local"
    auto_create_layers: bool = True
    paths: Dict[str, str] = field(default_factory=lambda: {
        "raw": "raw",
        "processed": "processed", 
        "curated": "curated",
        "metadata": "metadata",
        "schemas": "schemas",
        "logs": "logs"
    })


@dataclass
class SemanticLayerConfig:
    """Semantic layer configuration"""
    ontology_path: str = "./schemas/clinical_ontology.ttl"
    semantic_model_path: str = "./schemas/semantic_model.json"
    terminology_mappings: Dict[str, str] = field(default_factory=lambda: {
        "snomed_ct": "http://snomed.info/sct",
        "loinc": "http://loinc.org",
        "ctcae": "http://ctep.cancer.gov/protocolDevelopment/electronic_applications/ctc.htm"
    })


@dataclass
class AgentsConfig:
    """AI agents configuration"""
    llm: Dict[str, Any] = field(default_factory=lambda: {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.1,
        "max_tokens": 2000
    })
    clinical_analyst: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "confidence_threshold": 0.7
    })
    safety_monitor: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "alert_threshold": 0.8
    })
    statistical_analyzer: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "significance_level": 0.05
    })
    quality_assessor: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "quality_threshold": 0.8
    })


@dataclass
class ProcessingConfig:
    """Data processing configuration"""
    cdisc: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "version": "3.3",
        "validate_compliance": True
    })
    quality: Dict[str, Any] = field(default_factory=lambda: {
        "completeness_threshold": 0.9,
        "accuracy_threshold": 0.95,
        "consistency_threshold": 0.9
    })
    batch_size: int = 1000
    max_workers: int = 4
    timeout_seconds: int = 300


@dataclass
class SecurityConfig:
    """Security configuration"""
    encryption: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "algorithm": "AES-256"
    })
    access_control: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "default_role": "analyst"
    })
    audit: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "log_level": "INFO"
    })


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    metrics: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "collection_interval": 60
    })
    alerts: Dict[str, Any] = field(default_factory=lambda: {
        "email_enabled": False,
        "webhook_enabled": False
    })
    health_checks: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "interval": 300
    })


@dataclass
class IntegrationsConfig:
    """Integration configuration"""
    database: Dict[str, Any] = field(default_factory=lambda: {
        "type": "sqlite",
        "connection_string": "sqlite:///data/datalake.db"
    })
    terminology_services: Dict[str, Any] = field(default_factory=lambda: {
        "snomed_ct": {
            "endpoint": "https://browser.ihtsdotools.org/snowstorm/snomed-ct",
            "api_key": None
        },
        "loinc": {
            "endpoint": "https://fhir.loinc.org/CodeSystem/loinc",
            "api_key": None
        }
    })


@dataclass
class DevelopmentConfig:
    """Development configuration"""
    debug: bool = False
    testing: bool = False
    mock_data: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    rotation: str = "1 day"
    retention: str = "30 days"


@dataclass
class Settings:
    """Main settings class"""
    data_lake: DataLakeConfig = field(default_factory=DataLakeConfig)
    semantic_layer: SemanticLayerConfig = field(default_factory=SemanticLayerConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    integrations: IntegrationsConfig = field(default_factory=IntegrationsConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


class ConfigManager:
    """Configuration manager for the data lake"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.settings = Settings()
        
        # Load configuration
        self.load_config()
        
        # Setup logging
        self._setup_logging()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        # Look for config in several locations
        possible_paths = [
            "./config/settings.yaml",
            "../config/settings.yaml",
            "../../config/settings.yaml",
            os.path.expanduser("~/.datalakeagents/settings.yaml"),
            "/etc/datalakeagents/settings.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return default path if none found
        return "./config/settings.yaml"
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Update settings with loaded data
                self._update_settings(config_data)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Configuration file not found at {self.config_path}, using defaults")
                self._create_default_config()
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
    
    def _update_settings(self, config_data: Dict[str, Any]):
        """Update settings with configuration data"""
        if "data_lake" in config_data:
            self._update_dataclass(self.settings.data_lake, config_data["data_lake"])
        
        if "semantic_layer" in config_data:
            self._update_dataclass(self.settings.semantic_layer, config_data["semantic_layer"])
        
        if "agents" in config_data:
            self._update_dataclass(self.settings.agents, config_data["agents"])
        
        if "processing" in config_data:
            self._update_dataclass(self.settings.processing, config_data["processing"])
        
        if "security" in config_data:
            self._update_dataclass(self.settings.security, config_data["security"])
        
        if "monitoring" in config_data:
            self._update_dataclass(self.settings.monitoring, config_data["monitoring"])
        
        if "integrations" in config_data:
            self._update_dataclass(self.settings.integrations, config_data["integrations"])
        
        if "development" in config_data:
            self._update_dataclass(self.settings.development, config_data["development"])
        
        if "logging" in config_data:
            self._update_dataclass(self.settings.logging, config_data["logging"])
    
    def _update_dataclass(self, dataclass_instance: Any, data: Dict[str, Any]):
        """Update dataclass instance with dictionary data"""
        for key, value in data.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
    
    def _create_default_config(self):
        """Create default configuration file"""
        try:
            config_dir = Path(self.config_path).parent
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Create default configuration
            default_config = {
                "data_lake": {
                    "base_path": "./data",
                    "storage_backend": "local",
                    "auto_create_layers": True
                },
                "semantic_layer": {
                    "ontology_path": "./schemas/clinical_ontology.ttl",
                    "semantic_model_path": "./schemas/semantic_model.json"
                },
                "agents": {
                    "llm": {
                        "provider": "openai",
                        "model": "gpt-4",
                        "temperature": 0.1
                    }
                },
                "logging": {
                    "level": "INFO",
                    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
                }
            }
            
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Created default configuration at {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error creating default configuration: {e}")
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        try:
            # Remove default logger
            logger.remove()
            
            # Add console logger
            logger.add(
                sink=lambda msg: print(msg, end=""),
                level=self.settings.logging.level,
                format=self.settings.logging.format
            )
            
            # Add file logger if logs directory exists
            logs_dir = Path(self.settings.data_lake.base_path) / self.settings.data_lake.paths["logs"]
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = logs_dir / "datalake.log"
            logger.add(
                sink=str(log_file),
                level=self.settings.logging.level,
                format=self.settings.logging.format,
                rotation=self.settings.logging.rotation,
                retention=self.settings.logging.retention
            )
            
        except Exception as e:
            logger.error(f"Error setting up logging: {e}")
    
    def get_setting(self, section: str, key: str, default: Any = None) -> Any:
        """Get a specific setting value"""
        try:
            section_obj = getattr(self.settings, section)
            return getattr(section_obj, key, default)
        except AttributeError:
            return default
    
    def set_setting(self, section: str, key: str, value: Any):
        """Set a specific setting value"""
        try:
            section_obj = getattr(self.settings, section)
            setattr(section_obj, key, value)
        except AttributeError:
            logger.error(f"Invalid section or key: {section}.{key}")
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to file"""
        try:
            output_path = output_path or self.config_path
            
            config_data = {
                "data_lake": self._dataclass_to_dict(self.settings.data_lake),
                "semantic_layer": self._dataclass_to_dict(self.settings.semantic_layer),
                "agents": self._dataclass_to_dict(self.settings.agents),
                "processing": self._dataclass_to_dict(self.settings.processing),
                "security": self._dataclass_to_dict(self.settings.security),
                "monitoring": self._dataclass_to_dict(self.settings.monitoring),
                "integrations": self._dataclass_to_dict(self.settings.integrations),
                "development": self._dataclass_to_dict(self.settings.development),
                "logging": self._dataclass_to_dict(self.settings.logging)
            }
            
            with open(output_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _dataclass_to_dict(self, dataclass_instance: Any) -> Dict[str, Any]:
        """Convert dataclass to dictionary"""
        if hasattr(dataclass_instance, '__dict__'):
            return {k: v for k, v in dataclass_instance.__dict__.items() if not k.startswith('_')}
        return {}
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate configuration and return any issues"""
        issues = {
            "errors": [],
            "warnings": []
        }
        
        # Validate data lake paths
        base_path = Path(self.settings.data_lake.base_path)
        if not base_path.exists():
            try:
                base_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues["errors"].append(f"Cannot create data lake base path: {e}")
        
        # Validate semantic layer paths
        ontology_path = Path(self.settings.semantic_layer.ontology_path)
        if not ontology_path.exists():
            ontology_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate agent configuration
        if self.settings.agents.llm["provider"] not in ["openai", "anthropic", "local"]:
            issues["warnings"].append(f"Unknown LLM provider: {self.settings.agents.llm['provider']}")
        
        # Validate processing configuration
        if self.settings.processing.batch_size <= 0:
            issues["errors"].append("Batch size must be positive")
        
        if self.settings.processing.max_workers <= 0:
            issues["errors"].append("Max workers must be positive")
        
        return issues


# Global configuration instance
config_manager = ConfigManager()
settings = config_manager.settings
