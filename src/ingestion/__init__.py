"""
Data Ingestion Pipelines

Provides automated data ingestion pipelines for clinical trial data
with validation, transformation, and metadata extraction.
"""

__version__ = "0.1.0"

from .pipeline import IngestionPipeline
from .validators import DataValidator, SchemaValidator
from .transformers import DataTransformer, CDISCTransformer
from .extractors import MetadataExtractor

__all__ = [
    "IngestionPipeline",
    "DataValidator",
    "SchemaValidator", 
    "DataTransformer",
    "CDISCTransformer",
    "MetadataExtractor"
]
