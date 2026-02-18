"""
Metadata Manager for Clinical Trials Data Lake

Manages metadata for datasets, schemas, and data lineage.
Provides semantic annotations and data governance capabilities.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import hashlib
from pathlib import Path
import pandas as pd
from loguru import logger


class DataQualityLevel(Enum):
    """Data quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


class ProcessingStatus(Enum):
    """Data processing status"""
    RAW = "raw"
    PROCESSING = "processing"
    PROCESSED = "processed"
    VALIDATED = "validated"
    CURATED = "curated"
    ERROR = "error"


@dataclass
class DatasetMetadata:
    """Metadata for a dataset"""
    dataset_id: str
    name: str
    description: str
    domain: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    file_path: str
    file_size: int
    file_format: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    schema_version: str = "1.0"
    data_quality_level: DataQualityLevel = DataQualityLevel.UNKNOWN
    processing_status: ProcessingStatus = ProcessingStatus.RAW
    tags: List[str] = field(default_factory=list)
    semantic_annotations: Dict[str, str] = field(default_factory=dict)
    lineage: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    access_level: str = "internal"  # internal, restricted, public
    retention_period: Optional[str] = None
    checksum: Optional[str] = None


@dataclass
class ColumnMetadata:
    """Metadata for a column"""
    column_name: str
    data_type: str
    description: str
    nullable: bool = True
    unique_values: Optional[int] = None
    null_count: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    semantic_concept_id: Optional[str] = None
    terminology_mapping: Optional[Dict[str, str]] = None
    validation_rules: List[str] = field(default_factory=list)
    data_quality_score: float = 1.0


class MetadataManager:
    """Manages metadata for clinical trial datasets"""
    
    def __init__(self, metadata_store_path: str = "data/metadata"):
        self.metadata_store_path = Path(metadata_store_path)
        self.metadata_store_path.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.datasets: Dict[str, DatasetMetadata] = {}
        self.schemas: Dict[str, Dict[str, ColumnMetadata]] = {}
        
        # Load existing metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from storage"""
        datasets_file = self.metadata_store_path / "datasets.json"
        schemas_file = self.metadata_store_path / "schemas.json"
        
        # Load datasets
        if datasets_file.exists():
            with open(datasets_file, 'r') as f:
                datasets_data = json.load(f)
                for dataset_id, data in datasets_data.items():
                    # Convert string booleans back to booleans
                    if 'cdisc_compliant' in data:
                        if data['cdisc_compliant'].lower() == 'true':
                            data['cdisc_compliant'] = True
                        else:
                            data['cdisc_compliant'] = False
                    
                    # Convert processing status
                    if 'processing_status' in data:
                        data['processing_status'] = ProcessingStatus(data['processing_status'])
                    
                    try:
                        metadata = DatasetMetadata(**data)
                        self.datasets[dataset_id] = metadata
                    except TypeError as e:
                        logger.warning(f"Error loading dataset {dataset_id}: {e}")
                        # Skip this dataset and continue
                        continue
        
        if schemas_file.exists():
            with open(schemas_file, 'r') as f:
                schemas_data = json.load(f)
                for dataset_id, columns_data in schemas_data.items():
                    columns = {}
                    for col_name, col_data in columns_data['columns']:
                        # Convert string booleans back to booleans
                        if 'nullable' in col_data:
                            col_data['nullable'] = col_data['nullable'].lower() == 'true'
                        
                        columns[col_name] = ColumnMetadata(**col_data)
                    self.schemas[dataset_id] = columns
        
        logger.info(f"Loaded metadata for {len(self.datasets)} datasets")
    
    def _save_metadata(self):
        """Save metadata to storage"""
        import numpy as np
        
        datasets_file = self.metadata_store_path / "datasets.json"
        schemas_file = self.metadata_store_path / "schemas.json"
        
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, bool):
                return str(obj)  # Convert bool to string for JSON
            return obj
        
        # Prepare datasets data
        datasets_data = {}
        for dataset_id, dataset in self.datasets.items():
            # Get the cdisc_compliant value safely
            cdisc_compliant_value = getattr(dataset, 'cdisc_compliant', False)
            
            data = {
                "dataset_id": dataset.dataset_id,
                "name": dataset.name,
                "description": dataset.description,
                "domain": dataset.domain,
                "file_path": dataset.file_path,
                "created_by": dataset.created_by,
                "created_at": dataset.created_at.isoformat(),
                "updated_at": dataset.updated_at.isoformat(),
                "row_count": dataset.row_count,
                "column_count": dataset.column_count,
                "data_quality_level": dataset.data_quality_level.value if dataset.data_quality_level else None,
                "tags": dataset.tags,
                "lineage": dataset.lineage,
                "access_level": dataset.access_level,
                "cdisc_compliant": str(cdisc_compliant_value),  # Convert bool to string
                "terminology_mappings": getattr(dataset, 'terminology_mappings', {})
            }
            datasets_data[dataset_id] = data
        
        with open(datasets_file, 'w') as f:
            json.dump(datasets_data, f, indent=2, default=convert_numpy_types)
        
        # Save schemas metadata
        schemas_data = {}
        for dataset_id, columns in self.schemas.items():
            schemas_data[dataset_id] = {
                "dataset_id": dataset_id,
                "columns": [
                    {
                        "column_name": col.column_name,
                        "data_type": col.data_type,
                        "description": col.description,
                        "nullable": str(col.nullable),  # Convert bool to string
                        "unique_values": convert_numpy_types(col.unique_values),
                        "null_count": convert_numpy_types(col.null_count),
                        "min_value": convert_numpy_types(col.min_value),
                        "max_value": convert_numpy_types(col.max_value),
                        "mean_value": convert_numpy_types(col.mean_value),
                        "std_value": convert_numpy_types(col.std_value),
                        "semantic_concept_id": col.semantic_concept_id,
                        "terminology_mapping": col.terminology_mapping,
                        "validation_rules": col.validation_rules,
                        "data_quality_score": col.data_quality_score
                    }
                    for col in columns.values()
                ]
            }
        
        with open(schemas_file, 'w') as f:
            json.dump(schemas_data, f, indent=2, default=convert_numpy_types)
        
        logger.info("Metadata saved to storage")
    
    def register_dataset(self, file_path: str, name: str, description: str, 
                        domain: str, created_by: str, **kwargs) -> str:
        """Register a new dataset with metadata"""
        # Generate dataset ID
        dataset_id = hashlib.md5(f"{file_path}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        # Get file information
        file_path_obj = Path(file_path)
        file_size = file_path_obj.stat().st_size if file_path_obj.exists() else 0
        file_format = file_path_obj.suffix.lower().lstrip('.')
        
        # Create metadata
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            name=name,
            description=description,
            domain=domain,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by=created_by,
            file_path=file_path,
            file_size=file_size,
            file_format=file_format,
            **kwargs
        )
        
        # Calculate checksum if file exists
        if file_path_obj.exists():
            metadata.checksum = self._calculate_checksum(file_path_obj)
        
        self.datasets[dataset_id] = metadata
        self._save_metadata()
        
        logger.info(f"Registered dataset: {name} ({dataset_id})")
        return dataset_id
    
    def analyze_dataset_schema(self, dataset_id: str, data: pd.DataFrame):
        """Analyze dataset schema and create column metadata"""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Update dataset metadata
        metadata = self.datasets[dataset_id]
        metadata.row_count = len(data)
        metadata.column_count = len(data.columns)
        metadata.updated_at = datetime.now()
        
        # Analyze columns
        columns = {}
        for column in data.columns:
            col_data = data[column]
            
            # Basic statistics
            null_count = col_data.isnull().sum()
            unique_values = col_data.nunique()
            
            # Handle both scalar and Series data with proper type checking
            try:
                max_value = col_data.max() if hasattr(col_data, 'max') else None
                mean_value = float(col_data.mean()) if hasattr(col_data, 'mean') and col_data.dtype in ['int64', 'float64'] else None
                std_value = float(col_data.std()) if hasattr(col_data, 'std') and col_data.dtype in ['int64', 'float64'] else None
                min_value = col_data.min() if hasattr(col_data, 'min') and col_data.dtype in ['int64', 'float64'] else None
            except (TypeError, ValueError):
                # Handle string/object columns
                max_value = None
                mean_value = None
                std_value = None
                min_value = None
            
            col_metadata = ColumnMetadata(
                column_name=column,
                data_type=str(col_data.dtype),
                description=f"Column {column}",
                nullable=null_count > 0,
                unique_values=unique_values,
                null_count=null_count,
                min_value=min_value,
                max_value=max_value,
                mean_value=mean_value,
                std_value=std_value
            )
            
            columns[column] = col_metadata
        
        self.schemas[dataset_id] = columns
        self._save_metadata()
        
        logger.info(f"Analyzed schema for dataset {dataset_id}")
    
    def add_semantic_annotation(self, dataset_id: str, column_name: str, 
                              semantic_concept_id: str, terminology_mapping: Optional[Dict] = None):
        """Add semantic annotation to a column"""
        if dataset_id not in self.schemas:
            raise ValueError(f"Schema for dataset {dataset_id} not found")
        
        if column_name not in self.schemas[dataset_id]:
            raise ValueError(f"Column {column_name} not found in dataset {dataset_id}")
        
        column_metadata = self.schemas[dataset_id][column_name]
        column_metadata.semantic_concept_id = semantic_concept_id
        if terminology_mapping:
            column_metadata.terminology_mapping = terminology_mapping
        
        # Update dataset semantic annotations
        if dataset_id in self.datasets:
            self.datasets[dataset_id].semantic_annotations[column_name] = semantic_concept_id
            self.datasets[dataset_id].updated_at = datetime.now()
        
        self._save_metadata()
        logger.info(f"Added semantic annotation for {dataset_id}.{column_name}")
    
    def update_data_quality(self, dataset_id: str, quality_level: DataQualityLevel, 
                           quality_metrics: Dict[str, float]):
        """Update data quality information"""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        metadata = self.datasets[dataset_id]
        metadata.data_quality_level = quality_level
        metadata.quality_metrics = quality_metrics
        metadata.updated_at = datetime.now()
        
        self._save_metadata()
        logger.info(f"Updated data quality for dataset {dataset_id}")
    
    def add_lineage(self, dataset_id: str, source_dataset_id: str):
        """Add lineage information"""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        metadata = self.datasets[dataset_id]
        if source_dataset_id not in metadata.lineage:
            metadata.lineage.append(source_dataset_id)
            metadata.updated_at = datetime.now()
            self._save_metadata()
        
        logger.info(f"Added lineage: {source_dataset_id} -> {dataset_id}")
    
    def update_processing_status(self, dataset_id: str, status: ProcessingStatus):
        """Update processing status"""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        metadata = self.datasets[dataset_id]
        metadata.processing_status = status
        metadata.updated_at = datetime.now()
        
        self._save_metadata()
        logger.info(f"Updated processing status for {dataset_id} to {status.value}")
    
    def search_datasets(self, query: str = "", domain: str = "", 
                       tags: List[str] = None, quality_level: DataQualityLevel = None) -> List[DatasetMetadata]:
        """Search datasets by various criteria"""
        results = []
        
        for metadata in self.datasets.values():
            # Filter by query (search in name and description)
            if query and query.lower() not in metadata.name.lower() and query.lower() not in metadata.description.lower():
                continue
            
            # Filter by domain
            if domain and metadata.domain != domain:
                continue
            
            # Filter by tags
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            
            # Filter by quality level
            if quality_level and metadata.data_quality_level != quality_level:
                continue
            
            results.append(metadata)
        
        return results
    
    def get_dataset_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get metadata for a specific dataset"""
        return self.datasets.get(dataset_id)
    
    def get_dataset_schema(self, dataset_id: str) -> Optional[Dict[str, ColumnMetadata]]:
        """Get schema metadata for a specific dataset"""
        return self.schemas.get(dataset_id)
    
    def get_lineage_graph(self, dataset_id: str) -> Dict[str, List[str]]:
        """Get lineage graph for a dataset"""
        lineage_graph = {}
        
        def build_lineage(ds_id, graph):
            if ds_id in self.datasets:
                metadata = self.datasets[ds_id]
                graph[ds_id] = metadata.lineage.copy()
                for source_id in metadata.lineage:
                    build_lineage(source_id, graph)
        
        build_lineage(dataset_id, lineage_graph)
        return lineage_graph
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def export_metadata(self, output_path: str):
        """Export all metadata to a file"""
        export_data = {
            "datasets": {
                dataset_id: asdict(metadata) 
                for dataset_id, metadata in self.datasets.items()
            },
            "schemas": {
                dataset_id: {
                    col_name: asdict(col_metadata)
                    for col_name, col_metadata in columns.items()
                }
                for dataset_id, columns in self.schemas.items()
            },
            "exported_at": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metadata exported to {output_path}")
