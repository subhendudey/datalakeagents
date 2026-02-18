"""
Agent Orchestrator

Coordinates multiple AI agents for comprehensive clinical trial data analysis.
Manages agent selection, execution order, and result aggregation.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from .base_agent import BaseAgent, AgentCapability, AgentContext, AgentResult
from .clinical_analyst import ClinicalDataAnalyst
from .safety_monitor import SafetyMonitoringAgent
from .statistical_analyzer import StatisticalAnalysisAgent
from .quality_assessor import DataQualityAgent


class AgentOrchestrator:
    """Orchestrates multiple AI agents for comprehensive analysis"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Initialize default agents
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize default set of agents"""
        self.register_agent("clinical_analyst", ClinicalDataAnalyst())
        # Note: Other agents would be initialized here when their classes are implemented
        # self.register_agent("safety_monitor", SafetyMonitoringAgent())
        # self.register_agent("statistical_analyzer", StatisticalAnalysisAgent())
        # self.register_agent("quality_assessor", DataQualityAgent())
    
    def register_agent(self, name: str, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agents[name] = agent
        logger.info(f"Registered agent: {name}")
    
    def unregister_agent(self, name: str):
        """Unregister an agent"""
        if name in self.agents:
            del self.agents[name]
            logger.info(f"Unregistered agent: {name}")
    
    def analyze_dataset(self, dataset_id: str, data: pd.DataFrame, metadata: Dict[str, Any],
                       semantic_model: Optional[Any] = None, user_query: Optional[str] = None,
                       preferred_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform comprehensive analysis using multiple agents"""
        
        # Create execution context
        context = AgentContext(
            dataset_id=dataset_id,
            data=data,
            metadata=metadata,
            semantic_model=semantic_model,
            user_query=user_query,
            execution_id=f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Select agents for execution
        selected_agents = self._select_agents(context, preferred_agents)
        
        if not selected_agents:
            logger.warning("No agents available for analysis")
            return {"success": False, "error": "No agents available"}
        
        # Execute agents
        execution_results = self._execute_agents(context, selected_agents)
        
        # Aggregate results
        aggregated_results = self._aggregate_results(execution_results, context)
        
        # Store execution history
        self._store_execution_history(context, execution_results, aggregated_results)
        
        return aggregated_results
    
    def _select_agents(self, context: AgentContext, preferred_agents: Optional[List[str]] = None) -> List[Tuple[str, BaseAgent]]:
        """Select appropriate agents for the context"""
        selected = []
        
        # If preferred agents are specified, try to use them
        if preferred_agents:
            for agent_name in preferred_agents:
                if agent_name in self.agents:
                    agent = self.agents[agent_name]
                    if agent.can_handle(context):
                        selected.append((agent_name, agent))
                    else:
                        logger.warning(f"Agent {agent_name} cannot handle this context")
        
        # If no preferred agents or none can handle, use automatic selection
        if not selected:
            for agent_name, agent in self.agents.items():
                if agent.can_handle(context):
                    selected.append((agent_name, agent))
        
        logger.info(f"Selected {len(selected)} agents for analysis: {[name for name, _ in selected]}")
        return selected
    
    def _execute_agents(self, context: AgentContext, selected_agents: List[Tuple[str, BaseAgent]]) -> Dict[str, AgentResult]:
        """Execute selected agents"""
        results = {}
        
        # Execute agents in parallel where possible
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all agent executions
            future_to_agent = {
                executor.submit(agent.execute, context): agent_name
                for agent_name, agent in selected_agents
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result()
                    results[agent_name] = result
                    logger.info(f"Agent {agent_name} completed: {result.success}")
                except Exception as e:
                    logger.error(f"Agent {agent_name} failed with exception: {e}")
                    results[agent_name] = AgentResult(
                        agent_name=agent_name,
                        capability=AgentCapability.DATA_ANALYSIS,
                        success=False,
                        error_message=str(e)
                    )
        
        return results
    
    def _aggregate_results(self, execution_results: Dict[str, AgentResult], context: AgentContext) -> Dict[str, Any]:
        """Aggregate results from multiple agents"""
        aggregated = {
            "execution_id": context.execution_id,
            "dataset_id": context.dataset_id,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "summary": {
                "total_agents": len(execution_results),
                "successful_agents": sum(1 for r in execution_results.values() if r.success),
                "failed_agents": sum(1 for r in execution_results.values() if not r.success)
            },
            "insights": [],
            "recommendations": [],
            "agent_results": {},
            "overall_confidence": 0.0,
            "execution_time": 0.0
        }
        
        # Collect insights and recommendations from all agents
        total_execution_time = 0
        confidence_scores = []
        
        for agent_name, result in execution_results.items():
            # Store individual agent results
            aggregated["agent_results"][agent_name] = {
                "success": result.success,
                "capability": result.capability.value,
                "insights": result.insights,
                "recommendations": result.recommendations,
                "confidence_score": result.confidence_score,
                "execution_time": result.execution_time,
                "error_message": result.error_message
            }
            
            # Aggregate insights
            if result.success:
                aggregated["insights"].extend(result.insights)
                aggregated["recommendations"].extend(result.recommendations)
                confidence_scores.append(result.confidence_score)
            
            total_execution_time += result.execution_time
        
        # Calculate overall confidence
        if confidence_scores:
            aggregated["overall_confidence"] = sum(confidence_scores) / len(confidence_scores)
        
        aggregated["execution_time"] = total_execution_time
        
        # Add data summary
        aggregated["data_summary"] = {
            "rows": len(context.data),
            "columns": len(context.data.columns),
            "missing_values": context.data.isnull().sum().sum(),
            "data_types": context.data.dtypes.value_counts().to_dict()
        }
        
        # Determine overall success
        aggregated["success"] = aggregated["summary"]["successful_agents"] > 0
        
        return aggregated
    
    def _store_execution_history(self, context: AgentContext, execution_results: Dict[str, AgentResult], 
                               aggregated_results: Dict[str, Any]):
        """Store execution history for analysis and debugging"""
        history_entry = {
            "execution_id": context.execution_id,
            "dataset_id": context.dataset_id,
            "timestamp": datetime.now().isoformat(),
            "context": {
                "sample_size": len(context.data),
                "columns": list(context.data.columns),
                "user_query": context.user_query
            },
            "execution_results": {
                agent_name: {
                    "success": result.success,
                    "capability": result.capability.value,
                    "confidence_score": result.confidence_score,
                    "execution_time": result.execution_time
                }
                for agent_name, result in execution_results.items()
            },
            "aggregated_results": {
                "total_insights": len(aggregated_results["insights"]),
                "total_recommendations": len(aggregated_results["recommendations"]),
                "overall_confidence": aggregated_results["overall_confidence"],
                "total_execution_time": aggregated_results["execution_time"]
            }
        }
        
        self.execution_history.append(history_entry)
        
        # Keep only last 100 executions to prevent memory issues
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def get_agent_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all agents"""
        stats = {}
        
        for agent_name, agent in self.agents.items():
            agent_stats = agent.get_execution_stats()
            stats[agent_name] = agent_stats
        
        # Add orchestrator-level stats
        if self.execution_history:
            total_executions = len(self.execution_history)
            successful_executions = sum(1 for entry in self.execution_history 
                                       if entry["aggregated_results"]["total_insights"] > 0)
            
            stats["orchestrator"] = {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
                "average_agents_per_execution": sum(
                    len(entry["execution_results"]) for entry in self.execution_history
                ) / total_executions if total_executions > 0 else 0
            }
        
        return stats
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:] if limit > 0 else self.execution_history
    
    def clear_execution_history(self):
        """Clear execution history"""
        self.execution_history.clear()
        logger.info("Execution history cleared")
    
    def get_available_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available agents"""
        agents_info = {}
        
        for agent_name, agent in self.agents.items():
            agents_info[agent_name] = {
                "name": agent.name,
                "capabilities": [cap.value for cap in agent.capabilities],
                "execution_count": len(agent.execution_history),
                "success_rate": agent.get_execution_stats().get("success_rate", 0)
            }
        
        return agents_info
