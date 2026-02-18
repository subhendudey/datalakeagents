"""
Clinical Trials Data Lake

A comprehensive data lake solution for storing, managing, and analyzing
clinical trial data with semantic metadata and AI-powered agents.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .data_lake import DataLakeStorage, RawLayer, ProcessedLayer, CuratedLayer
from .meta_layer import SemanticModel, ClinicalOntology, MetadataManager
from .agents import ClinicalDataAnalyst, AgentOrchestrator

__all__ = [
    "DataLakeStorage",
    "RawLayer",
    "ProcessedLayer", 
    "CuratedLayer",
    "SemanticModel",
    "ClinicalOntology",
    "MetadataManager",
    "ClinicalDataAnalyst",
    "AgentOrchestrator"
]
