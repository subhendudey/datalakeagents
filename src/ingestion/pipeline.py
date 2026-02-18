"""
Data Ingestion Pipeline

Orchestrates the complete data ingestion process from raw files to curated datasets.
Includes validation, transformation, quality assessment, and metadata management.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
from loguru import logger

from ..data_lake import DataLakeStorage, RawLayer, ProcessedLayer, CuratedLayer
from ..meta_layer import MetadataManager, SemanticModel
from .validators import DataValidator, SchemaValidator
from .transformers import DataTransformer, CDISCTransformer
from .extractors import MetadataExtractor


class IngestionStatus(Enum):
    """Ingestion pipeline status"""
    PENDING = "pending"
    VALIDATING = "validating"
    TRANSFORMING = "transforming"
    PROCESSING = "processing"
    CURATING = "curating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestionConfig:
    """Configuration for ingestion pipeline"""
    validate_schema: bool = True
    apply_cdisc_transforms: bool = True
    assess_quality: bool = True
    extract_metadata: bool = True
    create_lineage: bool = True
    skip_on_error: bool = False
    batch_size: int = 1000


@dataclass
class IngestionResult:
    """Result of ingestion pipeline execution"""
    status: IngestionStatus
    dataset_id: str
    source_file: str
    records_processed: int
    records_rejected: int
    validation_errors: List[str]
    transformation_warnings: List[str]
    quality_score: float
    processing_time: float
    created_datasets: List[str]
    error_message: Optional[str] = None


class IngestionPipeline:
    """Main data ingestion pipeline"""
    
    def __init__(self, 
                 storage: DataLakeStorage,
                 metadata_manager: MetadataManager,
                 semantic_model: Optional[SemanticModel] = None,
                 config: Optional[IngestionConfig] = None):
        
        self.storage = storage
        self.metadata_manager = metadata_manager
        self.semantic_model = semantic_model
        self.config = config or IngestionConfig()
        
        # Initialize components
        self.raw_layer = RawLayer(storage)
        self.processed_layer = ProcessedLayer(storage)
        self.curated_layer = CuratedLayer(storage)
        
        self.data_validator = DataValidator()
        self.schema_validator = SchemaValidator()
        self.data_transformer = DataTransformer()
        self.cdisc_transformer = CDISCTransformer()
        self.metadata_extractor = MetadataExtractor()
    
    def ingest_file(self, 
                   file_path: str,
                   dataset_name: str,
                   dataset_description: str,
                   domain: str,
                   created_by: str,
                   **kwargs) -> IngestionResult:
        """Ingest a single file through the complete pipeline"""
        
        start_time = datetime.now()
        logger.info(f"Starting ingestion of {file_path}")
        
        try:
            # Initialize result
            result = IngestionResult(
                status=IngestionStatus.PENDING,
                dataset_id="",
                source_file=file_path,
                records_processed=0,
                records_rejected=0,
                validation_errors=[],
                transformation_warnings=[],
                quality_score=0.0,
                processing_time=0.0,
                created_datasets=[]
            )
            
            # Step 1: Load raw data
            logger.info("Loading raw data...")
            raw_data = self._load_data(file_path)
            result.records_processed = len(raw_data)
            
            # Step 2: Register dataset in metadata
            logger.info("Registering dataset...")
            dataset_id = self.metadata_manager.register_dataset(
                file_path=file_path,
                name=dataset_name,
                description=dataset_description,
                domain=domain,
                created_by=created_by,
                **kwargs
            )
            result.dataset_id = dataset_id
            
            # Step 3: Validate data
            if self.config.validate_schema:
                logger.info("Validating data...")
                result.status = IngestionStatus.VALIDATING
                validation_result = self._validate_data(raw_data, domain)
                
                if not validation_result["valid"]:
                    result.validation_errors = validation_result["errors"]
                    if not self.config.skip_on_error:
                        result.status = IngestionStatus.FAILED
                        result.error_message = "Data validation failed"
                        return result
                
                result.records_rejected = validation_result.get("rejected_records", 0)
            
            # Step 4: Store in raw layer
            logger.info("Storing in raw layer...")
            raw_path = self.raw_layer.ingest_clinical_trial(
                trial_id=dataset_id,
                data=raw_data,
                source_format=Path(file_path).suffix.lower().lstrip('.')
            )
            result.created_datasets.append(raw_path)
            
            # Step 5: Transform data
            logger.info("Transforming data...")
            result.status = IngestionStatus.TRANSFORMING
            
            # Apply basic transformations
            processed_data = self.data_transformer.transform(raw_data)
            
            # Apply CDISC transformations if enabled
            if self.config.apply_cdisc_transforms:
                processed_data, warnings = self.cdisc_transformer.transform(processed_data, domain)
                result.transformation_warnings.extend(warnings)
            
            # Step 6: Store in processed layer
            logger.info("Storing in processed layer...")
            processed_path = self._store_processed_data(dataset_id, processed_data, domain)
            result.created_datasets.append(processed_path)
            
            # Step 7: Extract and analyze metadata
            if self.config.extract_metadata:
                logger.info("Extracting metadata...")
                self.metadata_manager.analyze_dataset_schema(dataset_id, processed_data)
                
                # Add semantic annotations if semantic model is available
                if self.semantic_model:
                    self._add_semantic_annotations(dataset_id, processed_data)
            
            # Step 8: Assess data quality
            if self.config.assess_quality:
                logger.info("Assessing data quality...")
                quality_score = self._assess_data_quality(processed_data)
                result.quality_score = quality_score
                
                # Update metadata with quality information
                from ..meta_layer.metadata_manager import DataQualityLevel
                quality_level = self._get_quality_level(quality_score)
                self.metadata_manager.update_data_quality(dataset_id, quality_level, {"overall_score": quality_score})
            
            # Step 9: Create curated dataset
            logger.info("Creating curated dataset...")
            result.status = IngestionStatus.CURATING
            curated_path = self.curated_layer.create_analytics_dataset([processed_path])
            result.created_datasets.append(curated_path)
            
            # Step 10: Update lineage
            if self.config.create_lineage:
                self._update_lineage(dataset_id, raw_path, processed_path, curated_path)
            
            # Mark as completed
            result.status = IngestionStatus.COMPLETED
            result.processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Ingestion completed successfully in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            result.status = IngestionStatus.FAILED
            result.error_message = str(e)
            result.processing_time = (datetime.now() - start_time).total_seconds()
            return result
    
    def ingest_batch(self, 
                    file_configs: List[Dict[str, Any]],
                    parallel: bool = False) -> List[IngestionResult]:
        """Ingest multiple files in batch"""
        
        results = []
        
        if parallel:
            # TODO: Implement parallel processing
            logger.warning("Parallel processing not yet implemented, using sequential processing")
        
        for config in file_configs:
            result = self.ingest_file(**config)
            results.append(result)
            
            # Stop on error if configured
            if result.status == IngestionStatus.FAILED and not self.config.skip_on_error:
                logger.error(f"Stopping batch ingestion due to error in {config['file_path']}")
                break
        
        return results
    
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file"""
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path_obj.suffix.lower()
        
        if suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix == '.parquet':
            return pd.read_parquet(file_path)
        elif suffix == '.json':
            return pd.read_json(file_path)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _validate_data(self, data: pd.DataFrame, domain: str) -> Dict[str, Any]:
        """Validate data against schema and business rules"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "rejected_records": 0
        }
        
        # Basic validation
        basic_result = self.data_validator.validate(data)
        if not basic_result["valid"]:
            validation_result["valid"] = False
            validation_result["errors"].extend(basic_result["errors"])
        
        # Schema validation
        schema_result = self.schema_validator.validate(data, domain)
        if not schema_result["valid"]:
            validation_result["valid"] = False
            validation_result["errors"].extend(schema_result["errors"])
        
        validation_result["warnings"].extend(basic_result.get("warnings", []))
        validation_result["warnings"].extend(schema_result.get("warnings", []))
        
        return validation_result
    
    def _store_processed_data(self, dataset_id: str, data: pd.DataFrame, domain: str) -> str:
        """Store processed data in the appropriate layer"""
        if domain.lower() == "demographics":
            return self.processed_layer.process_demographics(f"raw/{dataset_id}.parquet")
        elif domain.lower() == "vital_signs":
            return self.processed_layer.process_vital_signs(f"raw/{dataset_id}.parquet")
        else:
            # Generic processing
            dataset_name = f"{dataset_id}_processed.parquet"
            return self.processed_layer.storage.write_processed(dataset_name, data, "parquet")
    
    def _add_semantic_annotations(self, dataset_id: str, data: pd.DataFrame):
        """Add semantic annotations to dataset columns"""
        if not self.semantic_model:
            return
        
        for column in data.columns:
            # Try to find matching concept
            concept = self.semantic_model.get_concept(column)
            if concept:
                terminology_mapping = None
                if concept.terminology and concept.terminology_code:
                    terminology_mapping = {
                        "terminology": concept.terminology,
                        "code": concept.terminology_code
                    }
                
                self.metadata_manager.add_semantic_annotation(
                    dataset_id, column, concept.concept_id, terminology_mapping
                )
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess overall data quality score"""
        quality_metrics = {}
        
        # Completeness (percentage of non-null values)
        completeness = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
        quality_metrics["completeness"] = completeness
        
        # Uniqueness (for ID columns)
        id_columns = [col for col in data.columns if 'id' in col.lower()]
        if id_columns:
            uniqueness_scores = []
            for col in id_columns:
                unique_ratio = data[col].nunique() / len(data)
                uniqueness_scores.append(unique_ratio)
            quality_metrics["uniqueness"] = sum(uniqueness_scores) / len(uniqueness_scores)
        else:
            quality_metrics["uniqueness"] = 1.0
        
        # Validity (basic range checks)
        validity_score = 1.0
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                # Check for negative values where inappropriate
                if any(keyword in col.lower() for keyword in ['age', 'weight', 'height', 'count']):
                    negative_count = (data[col] < 0).sum()
                    if negative_count > 0:
                        validity_score -= (negative_count / len(data)) * 0.1
        
        quality_metrics["validity"] = max(0.0, validity_score)
        
        # Overall quality score (weighted average)
        weights = {"completeness": 0.4, "uniqueness": 0.3, "validity": 0.3}
        overall_score = sum(quality_metrics[metric] * weight for metric, weight in weights.items())
        
        return round(overall_score, 2)
    
    def _get_quality_level(self, score: float) -> str:
        """Convert quality score to quality level"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.6:
            return "fair"
        else:
            return "poor"
    
    def _update_lineage(self, dataset_id: str, raw_path: str, processed_path: str, curated_path: str):
        """Update data lineage information"""
        # Add lineage relationships
        self.metadata_manager.add_lineage(processed_path.replace("processed/", ""), raw_path.replace("raw/", ""))
        self.metadata_manager.add_lineage(curated_path.replace("curated/", ""), processed_path.replace("processed/", ""))
