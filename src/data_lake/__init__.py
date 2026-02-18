"""
Clinical Trials Data Lake

A comprehensive data lake solution for storing, managing, and analyzing
clinical trial data with semantic metadata and AI-powered agents.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .storage import DataLakeStorage
from .layers import RawLayer, ProcessedLayer, CuratedLayer
from .catalog import DataCatalog

__all__ = [
    "DataLakeStorage",
    "RawLayer", 
    "ProcessedLayer",
    "CuratedLayer",
    "DataCatalog"
]
