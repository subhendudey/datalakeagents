"""
Meta Layer for Clinical Trials Data Lake

Provides semantic definitions, ontologies, and metadata management
for clinical trial data using RDF/OWL and CDISC standards.
"""

__version__ = "0.1.0"

from .semantic_model import SemanticModel, ClinicalOntology
from .metadata_manager import MetadataManager
from .cdisc_standards import CDISCStandards
from .data_catalog import DataCatalog

__all__ = [
    "SemanticModel",
    "ClinicalOntology", 
    "MetadataManager",
    "CDISCStandards",
    "DataCatalog"
]
