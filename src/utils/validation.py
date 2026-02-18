"""
Validation Utilities

Provides validation functions for clinical data and schemas.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger


def validate_clinical_data(data: pd.DataFrame, domain: str) -> Dict[str, Any]:
    """Validate clinical data against domain-specific rules"""
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "score": 1.0
    }
    
    # Basic validation
    if data.empty:
        validation_result["valid"] = False
        validation_result["errors"].append("Dataset is empty")
        return validation_result
    
    # Domain-specific validation
    if domain.lower() == "demographics":
        validation_result.update(_validate_demographics(data))
    elif domain.lower() == "vital_signs":
        validation_result.update(_validate_vital_signs(data))
    elif domain.lower() == "laboratory":
        validation_result.update(_validate_laboratory(data))
    elif domain.lower() == "adverse_events":
        validation_result.update(_validate_adverse_events(data))
    
    # Calculate overall score
    error_count = len(validation_result["errors"])
    warning_count = len(validation_result["warnings"])
    
    # Simple scoring: more errors/warnings = lower score
    validation_result["score"] = max(0.0, 1.0 - (error_count * 0.2 + warning_count * 0.05))
    
    return validation_result


def validate_schema(data: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data against a schema definition"""
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "missing_columns": [],
        "extra_columns": [],
        "type_mismatches": []
    }
    
    # Check required columns
    required_columns = schema.get("required_columns", [])
    for col in required_columns:
        if col not in data.columns:
            validation_result["missing_columns"].append(col)
            validation_result["valid"] = False
    
    # Check for extra columns
    allowed_columns = schema.get("allowed_columns", list(data.columns))
    extra_cols = [col for col in data.columns if col not in allowed_columns]
    if extra_cols:
        validation_result["extra_columns"] = extra_cols
        validation_result["warnings"].append(f"Extra columns found: {extra_cols}")
    
    # Check data types
    expected_types = schema.get("column_types", {})
    for col, expected_type in expected_types.items():
        if col in data.columns:
            actual_type = str(data[col].dtype)
            if not _is_type_compatible(actual_type, expected_type):
                validation_result["type_mismatches"].append({
                    "column": col,
                    "expected": expected_type,
                    "actual": actual_type
                })
                validation_result["valid"] = False
    
    return validation_result


def _validate_demographics(data: pd.DataFrame) -> Dict[str, Any]:
    """Validate demographics data"""
    errors = []
    warnings = []
    
    # Check for required columns
    required_cols = ['patient_id', 'age', 'gender']
    for col in required_cols:
        if col not in data.columns:
            errors.append(f"Missing required column: {col}")
    
    # Validate age
    if 'age' in data.columns:
        if data['age'].dtype in ['int64', 'float64']:
            invalid_ages = data[(data['age'] < 0) | (data['age'] > 150)]
            if len(invalid_ages) > 0:
                errors.append(f"Invalid age values: {len(invalid_ages)} records")
        else:
            errors.append("Age column should be numeric")
    
    # Validate gender
    if 'gender' in data.columns:
        valid_genders = ['M', 'F', 'Male', 'Female', 'U', 'Unknown']
        invalid_genders = data[~data['gender'].isin(valid_genders)]
        if len(invalid_genders) > 0:
            warnings.append(f"Unusual gender values: {len(invalid_genders)} records")
    
    return {"errors": errors, "warnings": warnings}


def _validate_vital_signs(data: pd.DataFrame) -> Dict[str, Any]:
    """Validate vital signs data"""
    errors = []
    warnings = []
    
    # Check for required columns
    required_cols = ['patient_id', 'measurement_date']
    for col in required_cols:
        if col not in data.columns:
            errors.append(f"Missing required column: {col}")
    
    # Validate blood pressure if present
    bp_columns = [col for col in data.columns if 'blood_pressure' in col.lower() or 'bp' in col.lower()]
    for col in bp_columns:
        if data[col].dtype in ['int64', 'float64']:
            # BP should be reasonable
            invalid_bp = data[(data[col] < 50) | (data[col] > 300)]
            if len(invalid_bp) > 0:
                warnings.append(f"Unusual blood pressure values in {col}: {len(invalid_bp)} records")
    
    return {"errors": errors, "warnings": warnings}


def _validate_laboratory(data: pd.DataFrame) -> Dict[str, Any]:
    """Validate laboratory data"""
    errors = []
    warnings = []
    
    # Check for required columns
    required_cols = ['patient_id', 'test_date', 'test_name', 'result']
    for col in required_cols:
        if col not in data.columns:
            errors.append(f"Missing required column: {col}")
    
    return {"errors": errors, "warnings": warnings}


def _validate_adverse_events(data: pd.DataFrame) -> Dict[str, Any]:
    """Validate adverse events data"""
    errors = []
    warnings = []
    
    # Check for required columns
    required_cols = ['patient_id', 'ae_term', 'start_date']
    for col in required_cols:
        if col not in data.columns:
            errors.append(f"Missing required column: {col}")
    
    return {"errors": errors, "warnings": warnings}


def _is_type_compatible(actual_type: str, expected_type: str) -> bool:
    """Check if actual data type is compatible with expected type"""
    type_mapping = {
        'string': ['object', 'string'],
        'integer': ['int64', 'int32'],
        'float': ['float64', 'float32', 'int64', 'int32'],
        'boolean': ['bool'],
        'date': ['datetime64[ns]', 'object'],
        'categorical': ['object']
    }
    
    compatible_types = type_mapping.get(expected_type, [])
    return actual_type in compatible_types
