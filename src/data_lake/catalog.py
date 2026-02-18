"""
Data Catalog for Data Lake

Provides cataloging and discovery capabilities for datasets
stored in the data lake.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from loguru import logger


@dataclass
class DatasetCatalogEntry:
    """Catalog entry for a dataset"""
    dataset_id: str
    name: str
    description: str
    layer: str  # raw, processed, curated
    format: str
    size_bytes: int
    record_count: int
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    metadata: Dict[str, Any]


class DataCatalog:
    """Data catalog for managing dataset metadata and discovery"""
    
    def __init__(self, catalog_path: str = "data/catalog"):
        self.catalog_path = Path(catalog_path)
        self.catalog_path.mkdir(parents=True, exist_ok=True)
        self.datasets: Dict[str, DatasetCatalogEntry] = {}
        
        # Load existing catalog
        self._load_catalog()
    
    def _load_catalog(self):
        """Load catalog from storage"""
        catalog_file = self.catalog_path / "catalog.json"
        if catalog_file.exists():
            try:
                with open(catalog_file, 'r') as f:
                    catalog_data = json.load(f)
                
                for dataset_id, entry_data in catalog_data.items():
                    # Convert datetime strings back to datetime objects
                    entry_data['created_at'] = datetime.fromisoformat(entry_data['created_at'])
                    entry_data['updated_at'] = datetime.fromisoformat(entry_data['updated_at'])
                    
                    self.datasets[dataset_id] = DatasetCatalogEntry(**entry_data)
                
                logger.info(f"Loaded catalog with {len(self.datasets)} datasets")
            except Exception as e:
                logger.error(f"Error loading catalog: {e}")
    
    def _save_catalog(self):
        """Save catalog to storage"""
        catalog_file = self.catalog_path / "catalog.json"
        
        catalog_data = {}
        for dataset_id, entry in self.datasets.items():
            entry_dict = {
                "dataset_id": entry.dataset_id,
                "name": entry.name,
                "description": entry.description,
                "layer": entry.layer,
                "format": entry.format,
                "size_bytes": entry.size_bytes,
                "record_count": entry.record_count,
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat(),
                "tags": entry.tags,
                "metadata": entry.metadata
            }
            catalog_data[dataset_id] = entry_dict
        
        with open(catalog_file, 'w') as f:
            json.dump(catalog_data, f, indent=2)
        
        logger.info(f"Saved catalog with {len(self.datasets)} datasets")
    
    def register_dataset(self, 
                      dataset_id: str,
                      name: str,
                      description: str,
                      layer: str,
                      format: str,
                      size_bytes: int,
                      record_count: int,
                      tags: List[str] = None,
                      metadata: Dict[str, Any] = None) -> DatasetCatalogEntry:
        """Register a new dataset in the catalog"""
        
        entry = DatasetCatalogEntry(
            dataset_id=dataset_id,
            name=name,
            description=description,
            layer=layer,
            format=format,
            size_bytes=size_bytes,
            record_count=record_count,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.datasets[dataset_id] = entry
        self._save_catalog()
        
        logger.info(f"Registered dataset {dataset_id} in catalog")
        return entry
    
    def update_dataset(self, dataset_id: str, **kwargs) -> Optional[DatasetCatalogEntry]:
        """Update an existing dataset entry"""
        if dataset_id not in self.datasets:
            return None
        
        entry = self.datasets[dataset_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        
        entry.updated_at = datetime.now()
        self._save_catalog()
        
        logger.info(f"Updated dataset {dataset_id} in catalog")
        return entry
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetCatalogEntry]:
        """Get a dataset entry by ID"""
        return self.datasets.get(dataset_id)
    
    def list_datasets(self, 
                   layer: Optional[str] = None,
                   tags: Optional[List[str]] = None,
                   limit: Optional[int] = None) -> List[DatasetCatalogEntry]:
        """List datasets with optional filtering"""
        
        datasets = list(self.datasets.values())
        
        # Filter by layer
        if layer:
            datasets = [d for d in datasets if d.layer == layer]
        
        # Filter by tags
        if tags:
            datasets = [d for d in datasets if any(tag in d.tags for tag in tags)]
        
        # Sort by updated date (newest first)
        datasets.sort(key=lambda x: x.updated_at, reverse=True)
        
        # Apply limit
        if limit:
            datasets = datasets[:limit]
        
        return datasets
    
    def search_datasets(self, query: str) -> List[DatasetCatalogEntry]:
        """Search datasets by name or description"""
        query_lower = query.lower()
        results = []
        
        for dataset in self.datasets.values():
            if (query_lower in dataset.name.lower() or 
                query_lower in dataset.description.lower()):
                results.append(dataset)
        
        return results
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset from the catalog"""
        if dataset_id in self.datasets:
            del self.datasets[dataset_id]
            self._save_catalog()
            logger.info(f"Deleted dataset {dataset_id} from catalog")
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get catalog statistics"""
        total_datasets = len(self.datasets)
        total_size = sum(d.size_bytes for d in self.datasets.values())
        total_records = sum(d.record_count for d in self.datasets.values())
        
        layer_counts = {}
        for dataset in self.datasets.values():
            layer_counts[dataset.layer] = layer_counts.get(dataset.layer, 0) + 1
        
        format_counts = {}
        for dataset in self.datasets.values():
            format_counts[dataset.format] = format_counts.get(dataset.format, 0) + 1
        
        return {
            "total_datasets": total_datasets,
            "total_size_bytes": total_size,
            "total_records": total_records,
            "datasets_by_layer": layer_counts,
            "datasets_by_format": format_counts
        }
