"""
Base Agent Framework

Provides the foundation for all AI agents in the clinical trials
data analysis system with common capabilities and interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import pandas as pd
from langchain.llms.base import LLM
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from loguru import logger


class AgentCapability(Enum):
    """Agent capabilities enumeration"""
    DATA_ANALYSIS = "data_analysis"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    SAFETY_MONITORING = "safety_monitoring"
    QUALITY_ASSESSMENT = "quality_assessment"
    SEMANTIC_UNDERSTANDING = "semantic_understanding"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    REPORT_GENERATION = "report_generation"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class AgentContext:
    """Context information for agent execution"""
    dataset_id: str
    data: pd.DataFrame
    metadata: Dict[str, Any]
    semantic_model: Optional[Any] = None
    user_query: Optional[str] = None
    execution_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentResult:
    """Result from agent execution"""
    agent_name: str
    capability: AgentCapability
    success: bool
    result_data: Any = None
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    execution_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, name: str, llm: Optional[LLM] = None, capabilities: List[AgentCapability] = None):
        self.name = name
        self.llm = llm
        self.capabilities = capabilities or []
        self.execution_history: List[AgentResult] = []
        
    @abstractmethod
    def can_handle(self, context: AgentContext) -> bool:
        """Check if agent can handle the given context"""
        pass
    
    @abstractmethod
    def execute(self, context: AgentContext) -> AgentResult:
        """Execute the agent's primary function"""
        pass
    
    def get_system_prompt(self) -> str:
        """Get system prompt for the agent"""
        return f"""
        You are {self.name}, a specialized AI agent for clinical trial data analysis.
        
        Your capabilities include: {', '.join([cap.value for cap in self.capabilities])}
        
        Always provide:
        1. Clear, actionable insights
        2. Evidence-based recommendations
        3. Confidence scores for your conclusions
        4. Identification of any limitations or assumptions
        
        Follow clinical data analysis best practices and maintain patient data privacy.
        """
    
    def validate_context(self, context: AgentContext) -> bool:
        """Validate the execution context"""
        if not context.dataset_id:
            logger.error("Missing dataset_id in context")
            return False
        
        if context.data is None or context.data.empty:
            logger.error("No data provided in context")
            return False
        
        return True
    
    def log_execution(self, result: AgentResult):
        """Log execution result"""
        self.execution_history.append(result)
        
        if result.success:
            logger.info(f"Agent {self.name} executed successfully in {result.execution_time:.2f}s")
        else:
            logger.error(f"Agent {self.name} failed: {result.error_message}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {"total_executions": 0}
        
        successful_executions = sum(1 for r in self.execution_history if r.success)
        avg_execution_time = sum(r.execution_time for r in self.execution_history) / len(self.execution_history)
        avg_confidence = sum(r.confidence_score for r in self.execution_history if r.success) / successful_executions if successful_executions > 0 else 0
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": successful_executions,
            "success_rate": successful_executions / len(self.execution_history),
            "average_execution_time": avg_execution_time,
            "average_confidence_score": avg_confidence
        }
    
    def _generate_insights_with_llm(self, context: AgentContext, analysis_prompt: str) -> List[str]:
        """Generate insights using LLM if available"""
        if not self.llm:
            return ["LLM not available for insight generation"]
        
        try:
            system_prompt = self.get_system_prompt()
            full_prompt = f"{system_prompt}\n\nContext: {context.user_query or 'Analyze the provided clinical trial data'}\n\n{analysis_prompt}"
            
            response = self.llm(full_prompt)
            return [response.strip()]
        except Exception as e:
            logger.error(f"Error generating insights with LLM: {e}")
            return ["Error generating AI-powered insights"]
    
    def _calculate_confidence_score(self, data_quality: float, sample_size: int, 
                                  result_consistency: float = 1.0) -> float:
        """Calculate confidence score for agent results"""
        # Base confidence from data quality
        base_confidence = data_quality
        
        # Adjust for sample size (larger samples increase confidence)
        size_factor = min(1.0, sample_size / 1000)  # Normalize to 0-1
        
        # Adjust for result consistency
        consistency_factor = result_consistency
        
        # Combined confidence score
        confidence = base_confidence * 0.4 + size_factor * 0.3 + consistency_factor * 0.3
        
        return round(confidence, 2)


class MultiCapabilityAgent(BaseAgent):
    """Agent that can handle multiple capabilities"""
    
    def __init__(self, name: str, llm: Optional[LLM] = None, capabilities: List[AgentCapability] = None):
        super().__init__(name, llm, capabilities)
        self.capability_handlers = {}
    
    def register_capability_handler(self, capability: AgentCapability, handler_func):
        """Register a handler function for a specific capability"""
        self.capability_handlers[capability] = handler_func
    
    def execute_capability(self, capability: AgentCapability, context: AgentContext) -> AgentResult:
        """Execute a specific capability"""
        if capability not in self.capability_handlers:
            return AgentResult(
                agent_name=self.name,
                capability=capability,
                success=False,
                error_message=f"No handler registered for capability {capability.value}"
            )
        
        handler = self.capability_handlers[capability]
        return handler(context)


class AgentCollaboration:
    """Manages collaboration between agents"""
    
    def __init__(self):
        self.collaboration_rules: Dict[str, List[str]] = {}
    
    def add_collaboration_rule(self, primary_agent: str, collaborating_agents: List[str]):
        """Add collaboration rule"""
        self.collaboration_rules[primary_agent] = collaborating_agents
    
    def get_collaborating_agents(self, agent_name: str) -> List[str]:
        """Get collaborating agents for a given agent"""
        return self.collaboration_rules.get(agent_name, [])
    
    def should_collaborate(self, context: AgentContext, primary_agent: BaseAgent) -> bool:
        """Determine if collaboration is needed"""
        # Simple heuristic: collaborate if data is complex or query is ambiguous
        if context.data.shape[1] > 20:  # Many columns
            return True
        
        if context.user_query and len(context.user_query.split()) > 20:  # Long query
            return True
        
        return False
