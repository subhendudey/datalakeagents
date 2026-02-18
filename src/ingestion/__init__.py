"""
Data Ingestion Pipelines

Provides automated data ingestion pipelines for clinical trial data
with validation, transformation, and metadata extraction.
"""

__version__ = "0.1.0"

from .pipeline import IngestionPipeline, IngestionConfig
from .validators import DataValidator, SchemaValidator
from .transformers import DataTransformer, CDISCTransformer
from .extractors import MetadataExtractor

__all__ = [
    "IngestionPipeline",
    "IngestionConfig",
    "DataValidator",
    "SchemaValidator", 
    "DataTransformer",
    "CDISCTransformer",
    "MetadataExtractor"
]
