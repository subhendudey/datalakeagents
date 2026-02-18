"""
AI Agents Framework for Clinical Trials Data Analysis

Provides specialized AI agents for analyzing clinical trial data
using semantic understanding and domain expertise.
"""

__version__ = "0.1.0"

from .base_agent import BaseAgent, AgentCapability
from .clinical_analyst import ClinicalDataAnalyst
from .safety_monitor import SafetyMonitoringAgent
from .statistical_analyzer import StatisticalAnalysisAgent
from .quality_assessor import DataQualityAgent
from .agent_orchestrator import AgentOrchestrator

__all__ = [
    "BaseAgent",
    "AgentCapability",
    "ClinicalDataAnalyst",
    "SafetyMonitoringAgent", 
    "StatisticalAnalysisAgent",
    "DataQualityAgent",
    "AgentOrchestrator"
]
