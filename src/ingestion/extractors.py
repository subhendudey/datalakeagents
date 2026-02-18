"""
Metadata Extractors for Ingestion Pipeline

Provides metadata extraction capabilities for clinical trial data
including data profiling and semantic analysis.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger


class MetadataExtractor:
    """Metadata extractor for datasets"""
    
    def extract_basic_metadata(self, data: pd.DataFrame, file_path: str) -> Dict[str, Any]:
        """Extract basic metadata from dataset"""
        metadata = {
            "file_path": file_path,
            "extraction_timestamp": datetime.now().isoformat(),
            "row_count": len(data),
            "column_count": len(data.columns),
            "columns": list(data.columns),
            "data_types": data.dtypes.to_dict(),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "missing_values": data.isnull().sum().to_dict(),
            "duplicate_rows": data.duplicated().sum()
        }
        
        # Add column statistics
        column_stats = {}
        for column in data.columns:
            stats = {
                "dtype": str(data[column].dtype),
                "non_null_count": data[column].count(),
                "null_count": data[column].isnull().sum(),
                "unique_count": data[column].nunique()
            }
            
            # Add numeric statistics
            if pd.api.types.is_numeric_dtype(data[column]):
                stats.update({
                    "min": data[column].min(),
                    "max": data[column].max(),
                    "mean": data[column].mean(),
                    "std": data[column].std(),
                    "median": data[column].median()
                })
            
            # Add text statistics
            elif pd.api.types.is_string_dtype(data[column]) or data[column].dtype == 'object':
                stats.update({
                    "avg_length": data[column].astype(str).str.len().mean(),
                    "max_length": data[column].astype(str).str.len().max(),
                    "min_length": data[column].astype(str).str.len().min()
                })
            
            column_stats[column] = stats
        
        metadata["column_statistics"] = column_stats
        
        return metadata
    
    def extract_semantic_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract semantic metadata from dataset"""
        semantic_metadata = {
            "clinical_domains": self._identify_clinical_domains(data),
            "data_quality": self._assess_data_quality(data),
            "temporal_coverage": self._extract_temporal_coverage(data),
            "patient_coverage": self._extract_patient_coverage(data)
        }
        
        return semantic_metadata
    
    def _identify_clinical_domains(self, data: pd.DataFrame) -> List[str]:
        """Identify clinical domains in the dataset"""
        domains = []
        columns = [col.lower() for col in data.columns]
        
        # Demographics indicators
        if any(indicator in ' '.join(columns) for indicator in ['age', 'gender', 'sex', 'weight', 'height', 'race', 'ethnic']):
            domains.append("demographics")
        
        # Vital signs indicators
        if any(indicator in ' '.join(columns) for indicator in ['blood_pressure', 'bp', 'heart_rate', 'temperature', 'pulse', 'respiratory']):
            domains.append("vital_signs")
        
        # Laboratory indicators
        if any(indicator in ' '.join(columns) for indicator in ['lab', 'laboratory', 'test', 'result', 'hemoglobin', 'creatinine', 'alt', 'ast']):
            domains.append("laboratory")
        
        # Adverse events indicators
        if any(indicator in ' '.join(columns) for indicator in ['adverse', 'ae', 'event', 'severity', 'serious']):
            domains.append("adverse_events")
        
        # Medication indicators
        if any(indicator in ' '.join(columns) for indicator in ['medication', 'drug', 'dose', 'administration', 'concomitant']):
            domains.append("medications")
        
        return domains
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics"""
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        
        # Check for duplicates
        duplicate_rate = data.duplicated().sum() / len(data) if len(data) > 0 else 0
        
        # Check for consistency in key columns
        id_columns = [col for col in data.columns if 'id' in col.lower()]
        consistency_score = 1.0
        for col in id_columns:
            if col in data.columns:
                unique_ratio = data[col].nunique() / len(data)
                consistency_score *= unique_ratio
        
        # Calculate overall quality score
        completeness_score = completeness
        duplicate_rate = duplicate_rate
        overall_quality_score = (completeness * 0.5 + (1 - duplicate_rate) * 0.3 + consistency_score * 0.2)
        
        return {
            "completeness_score": completeness,
            "duplicate_rate": duplicate_rate,
            "consistency_score": consistency_score,
            "overall_quality_score": overall_quality_score
        }
    
    def _extract_temporal_coverage(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract temporal coverage information"""
        date_columns = []
        for col in data.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time']):
                try:
                    # Try to convert to datetime
                    pd.to_datetime(data[col], errors='coerce')
                    date_columns.append(col)
                except:
                    pass
        
        if not date_columns:
            return {"date_columns_found": False}
        
        temporal_info = {
            "date_columns_found": True,
            "date_columns": date_columns
        }
        
        # Extract date ranges for each date column
        for col in date_columns:
            try:
                dates = pd.to_datetime(data[col], errors='coerce').dropna()
                if not dates.empty:
                    temporal_info[f"{col}_range"] = {
                        "start": dates.min().isoformat(),
                        "end": dates.max().isoformat(),
                        "span_days": (dates.max() - dates.min()).days
                    }
            except:
                temporal_info[f"{col}_range"] = {"error": "Could not parse dates"}
        
        return temporal_info
    
    def _extract_patient_coverage(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract patient coverage information"""
        patient_columns = [col for col in data.columns if 'patient' in col.lower() or 'subject' in col.lower()]
        
        if not patient_columns:
            return {"patient_columns_found": False}
        
        patient_info = {
            "patient_columns_found": True,
            "patient_columns": patient_columns
        }
        
        for col in patient_columns:
            if col in data.columns:
                unique_patients = data[col].nunique()
                patient_info[f"{col}_unique_patients"] = unique_patients
                patient_info[f"{col}_records_per_patient"] = len(data) / unique_patients if unique_patients > 0 else 0
        
        return patient_info
