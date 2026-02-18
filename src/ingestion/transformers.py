"""
Data Transformers for Ingestion Pipeline

Provides data transformation capabilities for clinical trial data
including standardization and CDISC compliance.
"""

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from datetime import datetime
from loguru import logger


class DataTransformer:
    """Basic data transformer"""
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply basic transformations"""
        transformed = data.copy()
        
        # Standardize column names
        transformed.columns = [col.upper().replace(' ', '_') for col in transformed.columns]
        
        # Convert date columns
        date_columns = [col for col in transformed.columns if 'DATE' in col or 'TIME' in col]
        for col in date_columns:
            if col in transformed.columns:
                try:
                    transformed[col] = pd.to_datetime(transformed[col], errors='coerce')
                except:
                    logger.warning(f"Could not convert {col} to datetime")
        
        # Remove duplicates
        initial_count = len(transformed)
        transformed = transformed.drop_duplicates()
        if len(transformed) < initial_count:
            logger.info(f"Removed {initial_count - len(transformed)} duplicate rows")
        
        # Handle missing values
        numeric_columns = transformed.select_dtypes(include=['number']).columns
        transformed[numeric_columns] = transformed[numeric_columns].fillna(0)
        
        categorical_columns = transformed.select_dtypes(include=['object']).columns
        transformed[categorical_columns] = transformed[categorical_columns].fillna('UNKNOWN')
        
        return transformed


class CDISCTransformer:
    """CDISC standards transformer"""
    
    def transform(self, data: pd.DataFrame, domain: str) -> Tuple[pd.DataFrame, List[str]]:
        """Transform data to CDISC format"""
        warnings = []
        transformed = data.copy()
        
        if domain.lower() == "demographics":
            transformed, domain_warnings = self._transform_demographics(transformed)
            warnings.extend(domain_warnings)
        
        elif domain.lower() == "vital_signs":
            transformed, domain_warnings = self._transform_vital_signs(transformed)
            warnings.extend(domain_warnings)
        
        return transformed, warnings
    
    def _transform_demographics(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Transform demographics to CDISC DM format"""
        warnings = []
        
        # Add CDISC required columns
        if 'STUDYID' not in data.columns:
            data['STUDYID'] = 'STUDY001'
            warnings.append("Added default STUDYID")
        
        if 'DOMAIN' not in data.columns:
            data['DOMAIN'] = 'DM'
        
        if 'USUBJID' not in data.columns:
            if 'PATIENT_ID' in data.columns:
                data['USUBJID'] = data['PATIENT_ID']
            elif 'PATIENTID' in data.columns:
                data['USUBJID'] = data['PATIENTID']
            else:
                warnings.append("Could not create USUBJID - missing patient identifier")
        
        if 'SUBJID' not in data.columns:
            if 'PATIENT_ID' in data.columns:
                data['SUBJID'] = data['PATIENT_ID']
        
        return data, warnings
    
    def _transform_vital_signs(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Transform vital signs to CDISC VS format"""
        warnings = []
        
        # Add CDISC required columns
        if 'STUDYID' not in data.columns:
            data['STUDYID'] = 'STUDY001'
            warnings.append("Added default STUDYID")
        
        if 'DOMAIN' not in data.columns:
            data['DOMAIN'] = 'VS'
        
        if 'USUBJID' not in data.columns:
            if 'PATIENT_ID' in data.columns:
                data['USUBJID'] = data['PATIENT_ID']
            else:
                warnings.append("Could not create USUBJID - missing patient identifier")
        
        # Add sequence number
        if 'VSSEQ' not in data.columns:
            data['VSSEQ'] = range(1, len(data) + 1)
        
        return data, warnings
