"""
Data Lake Layers Implementation

Defines the three layers of the data lake:
- Raw: Ingested clinical trial data in original format
- Processed: Cleaned and standardized data
- Curated: Business-ready data with semantic annotations
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from datetime import datetime
from loguru import logger

from .storage import DataLakeStorage


class DataLayer(ABC):
    """Abstract base class for data lake layers"""
    
    def __init__(self, storage: DataLakeStorage, layer_name: str):
        self.storage = storage
        self.layer_name = layer_name
    
    @abstractmethod
    def validate_schema(self, data: Any, schema: Dict) -> bool:
        """Validate data against schema"""
        pass
    
    @abstractmethod
    def transform(self, data: Any, **kwargs) -> Any:
        """Transform data for this layer"""
        pass


class RawLayer(DataLayer):
    """Raw data layer - stores clinical trial data in original format"""
    
    def __init__(self, storage: DataLakeStorage):
        super().__init__(storage, "raw")
    
    def validate_schema(self, data: Any, schema: Dict) -> bool:
        """Basic validation for raw data"""
        if isinstance(data, pd.DataFrame):
            # Check if required columns exist
            required_columns = schema.get("required_columns", [])
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}")
                return False
        return True
    
    def transform(self, data: Any, **kwargs) -> Any:
        """Raw layer typically doesn't transform data"""
        return data
    
    def ingest_clinical_trial(self, trial_id: str, data: pd.DataFrame, 
                            source_format: str = "csv") -> str:
        """Ingest clinical trial data"""
        dataset_name = f"clinical_trial_{trial_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if not self.validate_schema(data, {"required_columns": ["patient_id"]}):
            raise ValueError("Invalid clinical trial data schema")
        
        path = self.storage.write_raw(f"{dataset_name}.{source_format}", data, source_format)
        logger.info(f"Ingested clinical trial {trial_id} to raw layer")
        return path
    
    def ingest_adverse_events(self, trial_id: str, data: pd.DataFrame) -> str:
        """Ingest adverse events data"""
        dataset_name = f"adverse_events_{trial_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        path = self.storage.write_raw(f"{dataset_name}.parquet", data, "parquet")
        logger.info(f"Ingested adverse events for trial {trial_id}")
        return path
    
    def ingest_laboratory_results(self, trial_id: str, data: pd.DataFrame) -> str:
        """Ingest laboratory results data"""
        dataset_name = f"lab_results_{trial_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        path = self.storage.write_raw(f"{dataset_name}.parquet", data, "parquet")
        logger.info(f"Ingested lab results for trial {trial_id}")
        return path


class ProcessedLayer(DataLayer):
    """Processed data layer - cleaned and standardized clinical data"""
    
    def __init__(self, storage: DataLakeStorage):
        super().__init__(storage, "processed")
    
    def validate_schema(self, data: Any, schema: Dict) -> bool:
        """Validate processed data against CDISC standards"""
        if isinstance(data, pd.DataFrame):
            # Check CDISC SDTM compliance
            required_columns = schema.get("required_columns", [])
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                logger.error(f"Missing CDISC required columns: {missing_columns}")
                return False
            
            # Check data types
            expected_types = schema.get("column_types", {})
            for col, expected_type in expected_types.items():
                if col in data.columns:
                    if not pd.api.types.is_dtype_equal(data[col].dtype, expected_type):
                        logger.warning(f"Column {col} has wrong type: {data[col].dtype} vs {expected_type}")
        
        return True
    
    def transform(self, data: Any, **kwargs) -> Any:
        """Transform raw data to processed format"""
        if isinstance(data, pd.DataFrame):
            # Standardize column names to CDISC format
            data.columns = [col.upper().replace(' ', '_') for col in data.columns]
            
            # Convert date columns
            date_columns = kwargs.get('date_columns', [])
            for col in date_columns:
                if col in data.columns:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
            
            # Remove duplicates
            data = data.drop_duplicates()
            
            # Handle missing values
            if kwargs.get('fill_missing', True):
                numeric_columns = data.select_dtypes(include=['number']).columns
                data[numeric_columns] = data[numeric_columns].fillna(0)
                
                categorical_columns = data.select_dtypes(include=['object']).columns
                data[categorical_columns] = data[categorical_columns].fillna('UNKNOWN')
        
        return data
    
    def process_demographics(self, raw_data_path: str) -> str:
        """Process demographics data to CDISC DM domain"""
        raw_data = self.storage.read_raw(raw_data_path)
        
        # Transform to CDISC DM format
        processed_data = self.transform(raw_data, date_columns=['birth_date', 'study_date'])
        
        # Add CDISC required columns
        processed_data['STUDYID'] = processed_data.get('study_id', 'UNKNOWN')
        processed_data['DOMAIN'] = 'DM'
        processed_data['USUBJID'] = processed_data['patient_id']
        processed_data['SUBJID'] = processed_data['patient_id']
        processed_data['RFICDTC'] = processed_data.get('informed_consent_date', '')
        processed_data['RFENDTC'] = processed_data.get('study_end_date', '')
        
        # Validate CDISC schema
        cdisc_dm_schema = {
            "required_columns": ["STUDYID", "DOMAIN", "USUBJID", "SUBJID"],
            "column_types": {"STUDYID": "object", "DOMAIN": "object", "USUBJID": "object"}
        }
        
        if not self.validate_schema(processed_data, cdisc_dm_schema):
            raise ValueError("Processed data doesn't meet CDISC DM schema")
        
        # Save to processed layer
        dataset_name = raw_data_path.split('/')[-1].replace('.parquet', '_processed.parquet')
        path = self.storage.write_processed(dataset_name, processed_data, "parquet")
        logger.info(f"Processed demographics data saved to {path}")
        return path
    
    def process_vital_signs(self, raw_data_path: str) -> str:
        """Process vital signs data to CDISC VS domain"""
        raw_data = self.storage.read_raw(raw_data_path)
        
        # Transform to CDISC VS format
        processed_data = self.transform(raw_data, date_columns=['measurement_date'])
        
        # Add CDISC required columns
        processed_data['STUDYID'] = processed_data.get('study_id', 'UNKNOWN')
        processed_data['DOMAIN'] = 'VS'
        processed_data['USUBJID'] = processed_data['patient_id']
        processed_data['VSTESTCD'] = processed_data.get('vital_sign_code', '')
        processed_data['VSTEST'] = processed_data.get('vital_sign_name', '')
        processed_data['VSORRES'] = processed_data.get('result_value', '')
        processed_data['VSORRESU'] = processed_data.get('unit', '')
        processed_data['VISIT'] = processed_data.get('visit_name', '')
        processed_data['VSDTC'] = processed_data.get('measurement_date', '')
        
        # Validate CDISC schema
        cdisc_vs_schema = {
            "required_columns": ["STUDYID", "DOMAIN", "USUBJID", "VSTESTCD", "VSTEST"],
            "column_types": {"STUDYID": "object", "DOMAIN": "object", "USUBJID": "object"}
        }
        
        if not self.validate_schema(processed_data, cdisc_vs_schema):
            raise ValueError("Processed data doesn't meet CDISC VS schema")
        
        # Save to processed layer
        dataset_name = raw_data_path.split('/')[-1].replace('.parquet', '_processed.parquet')
        path = self.storage.write_processed(dataset_name, processed_data, "parquet")
        logger.info(f"Processed vital signs data saved to {path}")
        return path


class CuratedLayer(DataLayer):
    """Curated data layer - business-ready data with semantic annotations"""
    
    def __init__(self, storage: DataLakeStorage):
        super().__init__(storage, "curated")
    
    def validate_schema(self, data: Any, schema: Dict) -> bool:
        """Validate curated data with semantic annotations"""
        if isinstance(data, pd.DataFrame):
            # Check semantic annotations exist
            required_metadata = schema.get("required_metadata", [])
            for metadata in required_metadata:
                if metadata not in data.attrs:
                    logger.error(f"Missing required metadata: {metadata}")
                    return False
        return True
    
    def transform(self, data: Any, **kwargs) -> Any:
        """Add semantic annotations and business logic"""
        if isinstance(data, pd.DataFrame):
            # Add semantic metadata
            data.attrs['semantic_layer'] = True
            data.attrs['created_at'] = datetime.now().isoformat()
            data.attrs['data_quality_score'] = kwargs.get('quality_score', 1.0)
            data.attrs['clinical_domain'] = kwargs.get('clinical_domain', 'unknown')
            
            # Add business calculations
            if 'age' in data.columns and 'weight' in data.columns:
                data['bmi'] = data['weight'] / ((data['age'] / 100) ** 2)
        
        return data
    
    def create_analytics_dataset(self, processed_data_paths: List[str]) -> str:
        """Create curated analytics dataset from multiple processed datasets"""
        all_data = []
        
        for path in processed_data_paths:
            data = self.storage.read_processed(path)
            all_data.append(data)
        
        # Combine datasets
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
        else:
            combined_data = pd.DataFrame()
        
        # Add semantic annotations
        curated_data = self.transform(
            combined_data, 
            clinical_domain="clinical_trials_analytics",
            quality_score=0.95
        )
        
        # Save to curated layer
        dataset_name = f"analytics_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        path = self.storage.write_curated(dataset_name, curated_data, "parquet")
        logger.info(f"Created analytics dataset: {path}")
        return path
