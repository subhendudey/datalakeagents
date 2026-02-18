"""
Data Catalog for Meta Layer

Provides cataloging and discovery capabilities for datasets
with semantic metadata and clinical domain information.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
from loguru import logger


@dataclass
class DatasetCatalogEntry:
    """Catalog entry for a dataset with semantic metadata"""
    dataset_id: str
    name: str
    description: str
    domain: str
    format: str
    size_bytes: int
    record_count: int
    clinical_domain: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    semantic_annotations: Dict[str, str] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    lineage: List[str] = field(default_factory=list)
    access_level: str = "internal"
    cdisc_compliant: bool = False
    terminology_mappings: Dict[str, Any] = field(default_factory=dict)


class DataCatalog:
    """Data catalog for managing dataset metadata and discovery with semantic capabilities"""
    
    def __init__(self, catalog_path: str = "data/catalog"):
        self.catalog_path = Path(catalog_path)
        self.catalog_path.mkdir(parents=True, exist_ok=True)
        self.datasets: Dict[str, DatasetCatalogEntry] = {}
        self.semantic_index: Dict[str, List[str]] = {}  # concept_id -> dataset_ids
        
        # Load existing catalog
        self._load_catalog()
    
    def _load_catalog(self):
        """Load catalog from storage"""
        catalog_file = self.catalog_path / "catalog.json"
        if catalog_file.exists():
            try:
                with open(catalog_file, 'r') as f:
                    catalog_data = json.load(f)
                
                for dataset_id, entry_data in catalog_data.get("datasets", {}).items():
                    # Convert datetime strings back to datetime objects
                    entry_data['created_at'] = datetime.fromisoformat(entry_data['created_at'])
                    entry_data['updated_at'] = datetime.fromisoformat(entry_data['updated_at'])
                    
                    self.datasets[dataset_id] = DatasetCatalogEntry(**entry_data)
                
                # Load semantic index
                self.semantic_index = catalog_data.get("semantic_index", {})
                
                logger.info(f"Loaded catalog with {len(self.datasets)} datasets")
            except Exception as e:
                logger.error(f"Error loading catalog: {e}")
    
    def _save_catalog(self):
        """Save catalog to storage"""
        catalog_file = self.catalog_path / "catalog.json"
        
        catalog_data = {
            "datasets": {},
            "semantic_index": self.semantic_index,
            "version": "1.0",
            "last_updated": datetime.now().isoformat()
        }
        
        for dataset_id, entry in self.datasets.items():
            entry_dict = {
                "dataset_id": entry.dataset_id,
                "name": entry.name,
                "description": entry.description,
                "domain": entry.domain,
                "clinical_domain": entry.clinical_domain,
                "format": entry.format,
                "size_bytes": entry.size_bytes,
                "record_count": entry.record_count,
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat(),
                "tags": entry.tags,
                "semantic_annotations": entry.semantic_annotations,
                "quality_metrics": entry.quality_metrics,
                "lineage": entry.lineage,
                "access_level": entry.access_level,
                "cdisc_compliant": entry.cdisc_compliant,
                "terminology_mappings": entry.terminology_mappings
            }
            catalog_data["datasets"][dataset_id] = entry_dict
        
        with open(catalog_file, 'w') as f:
            json.dump(catalog_data, f, indent=2)
        
        logger.info(f"Saved catalog with {len(self.datasets)} datasets")
    
    def register_dataset(self, 
                      dataset_id: str,
                      name: str,
                      description: str,
                      domain: str,
                      format: str,
                      size_bytes: int,
                      record_count: int,
                      clinical_domain: Optional[str] = None,
                      tags: List[str] = None,
                      semantic_annotations: Dict[str, str] = None,
                      terminology_mappings: Dict[str, Any] = None,
                      cdisc_compliant: bool = False) -> DatasetCatalogEntry:
        """Register a new dataset in the catalog"""
        
        entry = DatasetCatalogEntry(
            dataset_id=dataset_id,
            name=name,
            description=description,
            domain=domain,
            clinical_domain=clinical_domain,
            format=format,
            size_bytes=size_bytes,
            record_count=record_count,
            tags=tags or [],
            semantic_annotations=semantic_annotations or {},
            terminology_mappings=terminology_mappings or {},
            cdisc_compliant=cdisc_compliant
        )
        
        self.datasets[dataset_id] = entry
        
        # Update semantic index
        if semantic_annotations:
            for column, concept_id in semantic_annotations.items():
                if concept_id not in self.semantic_index:
                    self.semantic_index[concept_id] = []
                if dataset_id not in self.semantic_index[concept_id]:
                    self.semantic_index[concept_id].append(dataset_id)
        
        self._save_catalog()
        
        logger.info(f"Registered dataset {dataset_id} in catalog")
        return entry
    
    def update_dataset(self, dataset_id: str, **kwargs) -> Optional[DatasetCatalogEntry]:
        """Update an existing dataset entry"""
        if dataset_id not in self.datasets:
            return None
        
        entry = self.datasets[dataset_id]
        
        # Update semantic index if annotations changed
        if "semantic_annotations" in kwargs:
            old_annotations = entry.semantic_annotations
            new_annotations = kwargs["semantic_annotations"]
            
            # Remove old semantic index entries
            for column, concept_id in old_annotations.items():
                if concept_id in self.semantic_index and dataset_id in self.semantic_index[concept_id]:
                    self.semantic_index[concept_id].remove(dataset_id)
                    if not self.semantic_index[concept_id]:
                        del self.semantic_index[concept_id]
            
            # Add new semantic index entries
            for column, concept_id in new_annotations.items():
                if concept_id not in self.semantic_index:
                    self.semantic_index[concept_id] = []
                if dataset_id not in self.semantic_index[concept_id]:
                    self.semantic_index[concept_id].append(dataset_id)
        
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
                   domain: Optional[str] = None,
                   clinical_domain: Optional[str] = None,
                   tags: Optional[List[str]] = None,
                   cdisc_compliant: Optional[bool] = None,
                   limit: Optional[int] = None) -> List[DatasetCatalogEntry]:
        """List datasets with optional filtering"""
        
        datasets = list(self.datasets.values())
        
        # Filter by domain
        if domain:
            datasets = [d for d in datasets if d.domain == domain]
        
        # Filter by clinical domain
        if clinical_domain:
            datasets = [d for d in datasets if d.clinical_domain == clinical_domain]
        
        # Filter by tags
        if tags:
            datasets = [d for d in datasets if any(tag in d.tags for tag in tags)]
        
        # Filter by CDISC compliance
        if cdisc_compliant is not None:
            datasets = [d for d in datasets if d.cdisc_compliant == cdisc_compliant]
        
        # Sort by updated date (newest first)
        datasets.sort(key=lambda x: x.updated_at, reverse=True)
        
        # Apply limit
        if limit:
            datasets = datasets[:limit]
        
        return datasets
    
    def search_datasets(self, query: str) -> List[DatasetCatalogEntry]:
        """Search datasets by name, description, or semantic concepts"""
        query_lower = query.lower()
        results = []
        
        # Text search
        for dataset in self.datasets.values():
            if (query_lower in dataset.name.lower() or 
                query_lower in dataset.description.lower() or
                query_lower in dataset.clinical_domain.lower() if dataset.clinical_domain else False):
                results.append(dataset)
        
        # Semantic search
        if query_lower in self.semantic_index:
            semantic_results = []
            for dataset_id in self.semantic_index[query_lower]:
                if dataset_id in self.datasets:
                    semantic_results.append(self.datasets[dataset_id])
            
            # Combine and deduplicate
            all_results = results + semantic_results
            seen = set()
            results = []
            for item in all_results:
                if item.dataset_id not in seen:
                    seen.add(item.dataset_id)
                    results.append(item)
        
        return results
    
    def find_datasets_by_concept(self, concept_id: str) -> List[DatasetCatalogEntry]:
        """Find datasets that contain a specific semantic concept"""
        if concept_id not in self.semantic_index:
            return []
        
        dataset_ids = self.semantic_index[concept_id]
        return [self.datasets[dataset_id] for dataset_id in dataset_ids if dataset_id in self.datasets]
    
    def get_related_datasets(self, dataset_id: str) -> List[DatasetCatalogEntry]:
        """Find datasets related to a given dataset based on semantic overlap"""
        if dataset_id not in self.datasets:
            return []
        
        target_dataset = self.datasets[dataset_id]
        target_concepts = set(target_dataset.semantic_annotations.values())
        
        related = []
        for other_id, other_dataset in self.datasets.items():
            if other_id == dataset_id:
                continue
            
            other_concepts = set(other_dataset.semantic_annotations.values())
            
            # Calculate semantic similarity (Jaccard index)
            if target_concepts and other_concepts:
                intersection = len(target_concepts.intersection(other_concepts))
                union = len(target_concepts.union(other_concepts))
                similarity = intersection / union if union > 0 else 0
                
                if similarity > 0.1:  # Threshold for relatedness
                    related.append((other_dataset, similarity))
        
        # Sort by similarity score
        related.sort(key=lambda x: x[1], reverse=True)
        
        return [dataset for dataset, _ in related]
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset from the catalog"""
        if dataset_id not in self.datasets:
            return False
        
        # Remove from semantic index
        entry = self.datasets[dataset_id]
        for concept_id in entry.semantic_annotations.values():
            if concept_id in self.semantic_index and dataset_id in self.semantic_index[concept_id]:
                self.semantic_index[concept_id].remove(dataset_id)
                if not self.semantic_index[concept_id]:
                    del self.semantic_index[concept_id]
        
        # Remove from datasets
        del self.datasets[dataset_id]
        self._save_catalog()
        
        logger.info(f"Deleted dataset {dataset_id} from catalog")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get catalog statistics"""
        total_datasets = len(self.datasets)
        total_size = sum(d.size_bytes for d in self.datasets.values())
        total_records = sum(d.record_count for d in self.datasets.values())
        
        domain_counts = {}
        for dataset in self.datasets.values():
            domain_counts[dataset.domain] = domain_counts.get(dataset.domain, 0) + 1
        
        clinical_domain_counts = {}
        for dataset in self.datasets.values():
            if dataset.clinical_domain:
                clinical_domain_counts[dataset.clinical_domain] = clinical_domain_counts.get(dataset.clinical_domain, 0) + 1
        
        format_counts = {}
        for dataset in self.datasets.values():
            format_counts[dataset.format] = format_counts.get(dataset.format, 0) + 1
        
        cdisc_compliant_count = sum(1 for d in self.datasets.values() if d.cdisc_compliant)
        
        return {
            "total_datasets": total_datasets,
            "total_size_bytes": total_size,
            "total_records": total_records,
            "datasets_by_domain": domain_counts,
            "datasets_by_clinical_domain": clinical_domain_counts,
            "datasets_by_format": format_counts,
            "cdisc_compliant_datasets": cdisc_compliant_count,
            "cdisc_compliance_rate": cdisc_compliant_count / total_datasets if total_datasets > 0 else 0,
            "semantic_concepts_indexed": len(self.semantic_index)
        }
    
    def export_catalog(self, output_path: str, format: str = "json"):
        """Export catalog to file"""
        if format == "json":
            catalog_data = {
                "datasets": {
                    dataset_id: {
                        "dataset_id": entry.dataset_id,
                        "name": entry.name,
                        "description": entry.description,
                        "domain": entry.domain,
                        "clinical_domain": entry.clinical_domain,
                        "format": entry.format,
                        "size_bytes": entry.size_bytes,
                        "record_count": entry.record_count,
                        "created_at": entry.created_at.isoformat(),
                        "updated_at": entry.updated_at.isoformat(),
                        "tags": entry.tags,
                        "semantic_annotations": entry.semantic_annotations,
                        "quality_metrics": entry.quality_metrics,
                        "lineage": entry.lineage,
                        "access_level": entry.access_level,
                        "cdisc_compliant": entry.cdisc_compliant,
                        "terminology_mappings": entry.terminology_mappings
                    }
                    for dataset_id, entry in self.datasets.items()
                },
                "semantic_index": self.semantic_index,
                "statistics": self.get_statistics(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(catalog_data, f, indent=2)
        
        elif format == "csv":
            # Export datasets as CSV
            datasets_data = []
            for entry in self.datasets.values():
                datasets_data.append({
                    "dataset_id": entry.dataset_id,
                    "name": entry.name,
                    "description": entry.description,
                    "domain": entry.domain,
                    "clinical_domain": entry.clinical_domain,
                    "format": entry.format,
                    "size_bytes": entry.size_bytes,
                    "record_count": entry.record_count,
                    "created_at": entry.created_at.isoformat(),
                    "updated_at": entry.updated_at.isoformat(),
                    "tags": ",".join(entry.tags),
                    "cdisc_compliant": entry.cdisc_compliant,
                    "access_level": entry.access_level
                })
            
            df = pd.DataFrame(datasets_data)
            df.to_csv(output_path, index=False)
        
        logger.info(f"Exported catalog to {output_path} in {format} format")
