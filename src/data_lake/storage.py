"""
Data Lake Storage Implementation

Provides unified storage interface across different storage backends
(local filesystem, Azure Blob, S3) with Delta Lake support.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from delta import DeltaTable
from loguru import logger


class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    def write(self, path: str, data: Any, format: str = "parquet") -> None:
        """Write data to storage"""
        pass
    
    @abstractmethod
    def read(self, path: str, format: str = "parquet") -> Any:
        """Read data from storage"""
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if path exists"""
        pass
    
    @abstractmethod
    def list_files(self, path: str) -> List[str]:
        """List files in directory"""
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage backend"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_full_path(self, path: str) -> Path:
        return self.base_path / path
    
    def write(self, path: str, data: Any, format: str = "parquet") -> None:
        full_path = self._get_full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "parquet":
            if isinstance(data, pd.DataFrame):
                data.to_parquet(full_path, index=False)
            elif isinstance(data, pa.Table):
                pq.write_table(data, full_path)
            else:
                raise ValueError(f"Unsupported data type for parquet: {type(data)}")
        elif format == "csv":
            if isinstance(data, pd.DataFrame):
                data.to_csv(full_path, index=False)
            else:
                raise ValueError(f"Unsupported data type for csv: {type(data)}")
        elif format == "json":
            if isinstance(data, (dict, list)):
                import json
                with open(full_path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported data type for json: {type(data)}")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def read(self, path: str, format: str = "parquet") -> Any:
        full_path = self._get_full_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        
        if format == "parquet":
            return pd.read_parquet(full_path)
        elif format == "csv":
            return pd.read_csv(full_path)
        elif format == "json":
            import json
            with open(full_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def exists(self, path: str) -> bool:
        return self._get_full_path(path).exists()
    
    def list_files(self, path: str) -> List[str]:
        full_path = self._get_full_path(path)
        if not full_path.exists():
            return []
        
        if full_path.is_file():
            return [str(full_path.relative_to(self.base_path))]
        
        files = []
        for file_path in full_path.rglob("*"):
            if file_path.is_file():
                files.append(str(file_path.relative_to(self.base_path)))
        return files


class DataLakeStorage:
    """Unified data lake storage with multiple layers"""
    
    def __init__(self, base_path: str, backend: Optional[StorageBackend] = None):
        self.base_path = base_path
        self.backend = backend or LocalStorage(base_path)
        
        # Initialize layers
        self.raw_path = "raw"
        self.processed_path = "processed" 
        self.curated_path = "curated"
        
        # Create layer directories
        for layer_path in [self.raw_path, self.processed_path, self.curated_path]:
            self.backend.write(f"{layer_path}/.gitkeep", "", "txt")
    
    def write_raw(self, dataset: str, data: Any, format: str = "parquet") -> str:
        """Write data to raw layer"""
        path = f"{self.raw_path}/{dataset}"
        self.backend.write(path, data, format)
        logger.info(f"Written raw data to {path}")
        return path
    
    def write_processed(self, dataset: str, data: Any, format: str = "parquet") -> str:
        """Write data to processed layer"""
        path = f"{self.processed_path}/{dataset}"
        self.backend.write(path, data, format)
        logger.info(f"Written processed data to {path}")
        return path
    
    def write_curated(self, dataset: str, data: Any, format: str = "parquet") -> str:
        """Write data to curated layer"""
        path = f"{self.curated_path}/{dataset}"
        self.backend.write(path, data, format)
        logger.info(f"Written curated data to {path}")
        return path
    
    def read_raw(self, dataset: str, format: str = "parquet") -> Any:
        """Read data from raw layer"""
        path = f"{self.raw_path}/{dataset}"
        return self.backend.read(path, format)
    
    def read_processed(self, dataset: str, format: str = "parquet") -> Any:
        """Read data from processed layer"""
        path = f"{self.processed_path}/{dataset}"
        return self.backend.read(path, format)
    
    def read_curated(self, dataset: str, format: str = "parquet") -> Any:
        """Read data from curated layer"""
        path = f"{self.curated_path}/{dataset}"
        return self.backend.read(path, format)
    
    def list_datasets(self, layer: Optional[str] = None) -> Dict[str, List[str]]:
        """List all datasets by layer"""
        layers = {
            "raw": self.raw_path,
            "processed": self.processed_path,
            "curated": self.curated_path
        }
        
        if layer:
            layers = {layer: layers[layer]}
        
        result = {}
        for layer_name, layer_path in layers.items():
            files = self.backend.list_files(layer_path)
            # Filter out .gitkeep files
            datasets = [f for f in files if not f.endswith('.gitkeep')]
            result[layer_name] = datasets
        
        return result
