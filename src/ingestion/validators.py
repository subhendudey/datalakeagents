"""
Data Validators for Ingestion Pipeline

Provides validation capabilities for clinical trial data
including schema validation and business rule checks.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from loguru import logger


class DataValidator:
    """Basic data validator"""
    
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate basic data quality"""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check for empty data
        if data.empty:
            result["valid"] = False
            result["errors"].append("Dataset is empty")
            return result
        
        # Check for duplicate rows
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            result["warnings"].append(f"Found {duplicates} duplicate rows")
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        high_missing = missing_counts[missing_counts > len(data) * 0.5]
        if not high_missing.empty:
            result["warnings"].append(f"Columns with >50% missing values: {list(high_missing.index)}")
        
        return result


class SchemaValidator:
    """Schema validator for clinical data"""
    
    def validate(self, data: pd.DataFrame, domain: str) -> Dict[str, Any]:
        """Validate data against expected schema"""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Basic schema checks based on domain
        if domain.lower() == "demographics":
            required_columns = ["patient_id", "age", "gender"]
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                result["valid"] = False
                result["errors"].append(f"Missing required columns: {missing_cols}")
        
        elif domain.lower() == "vital_signs":
            required_columns = ["patient_id", "measurement_date"]
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                result["valid"] = False
                result["errors"].append(f"Missing required columns: {missing_cols}")
        
        return result
