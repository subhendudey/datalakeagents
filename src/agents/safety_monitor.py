"""
Safety Monitoring Agent

Specialized agent for monitoring clinical trial safety data
including adverse events and safety signals.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from .base_agent import BaseAgent, AgentCapability, AgentContext, AgentResult


class SafetyMonitoringAgent(BaseAgent):
    """Agent specialized in safety monitoring"""
    
    def __init__(self, llm=None):
        super().__init__(
            name="Safety Monitor",
            llm=llm,
            capabilities=[
                AgentCapability.SAFETY_MONITORING,
                AgentCapability.ANOMALY_DETECTION,
                AgentCapability.REPORT_GENERATION
            ]
        )
    
    def can_handle(self, context: AgentContext) -> bool:
        """Check if agent can handle the context"""
        # Check if data contains safety-related information
        safety_columns = self._identify_safety_columns(context.data)
        return len(safety_columns) > 0
    
    def execute(self, context: AgentContext) -> AgentResult:
        """Execute safety monitoring analysis"""
        start_time = datetime.now()
        
        try:
            if not self.validate_context(context):
                return AgentResult(
                    agent_name=self.name,
                    capability=AgentCapability.SAFETY_MONITORING,
                    success=False,
                    error_message="Invalid context provided"
                )
            
            # Identify safety data
            safety_columns = self._identify_safety_columns(context.data)
            
            if not safety_columns:
                return AgentResult(
                    agent_name=self.name,
                    capability=AgentCapability.SAFETY_MONITORING,
                    success=False,
                    error_message="No safety-related data found"
                )
            
            # Perform safety analysis
            insights = []
            recommendations = []
            result_data = {}
            
            # Analyze adverse events
            if 'adverse_events' in safety_columns:
                ae_insights, ae_recommendations, ae_results = self._analyze_adverse_events(context.data)
                insights.extend(ae_insights)
                recommendations.extend(ae_recommendations)
                result_data['adverse_events'] = ae_results
            
            # Analyze laboratory safety
            if 'laboratory' in safety_columns:
                lab_insights, lab_recommendations, lab_results = self._analyze_laboratory_safety(context.data)
                insights.extend(lab_insights)
                recommendations.extend(lab_recommendations)
                result_data['laboratory_safety'] = lab_results
            
            # Calculate confidence score
            confidence = self._calculate_safety_confidence(context, safety_columns)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = AgentResult(
                agent_name=self.name,
                capability=AgentCapability.SAFETY_MONITORING,
                success=True,
                result_data=result_data,
                insights=insights,
                recommendations=recommendations,
                confidence_score=confidence,
                execution_time=execution_time,
                metadata={
                    "safety_columns": safety_columns,
                    "sample_size": len(context.data)
                }
            )
            
            self.log_execution(result)
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in safety monitoring: {e}")
            
            return AgentResult(
                agent_name=self.name,
                capability=AgentCapability.SAFETY_MONITORING,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _identify_safety_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify safety-related columns"""
        safety_columns = []
        columns = data.columns.tolist()
        
        # Adverse events indicators
        if any(keyword in ' '.join(columns).lower() for keyword in ['adverse', 'ae', 'event', 'severity']):
            safety_columns.append('adverse_events')
        
        # Laboratory indicators
        if any(keyword in ' '.join(columns).lower() for keyword in ['lab', 'laboratory', 'test', 'result']):
            safety_columns.append('laboratory')
        
        # Vital signs indicators
        if any(keyword in ' '.join(columns).lower() for keyword in ['vital', 'bp', 'heart_rate', 'temperature']):
            safety_columns.append('vital_signs')
        
        return safety_columns
    
    def _analyze_adverse_events(self, data: pd.DataFrame) -> tuple:
        """Analyze adverse events data"""
        insights = []
        recommendations = []
        results = {}
        
        # Look for AE-related columns
        ae_columns = [col for col in data.columns if 'ae' in col.lower() or 'adverse' in col.lower()]
        
        if not ae_columns:
            insights.append("No adverse event data identified")
            return insights, recommendations, results
        
        # Analyze severity distribution
        severity_col = None
        for col in ae_columns:
            if 'severity' in col.lower() or 'grade' in col.lower():
                severity_col = col
                break
        
        if severity_col and severity_col in data.columns:
            severity_dist = data[severity_col].value_counts()
            results['severity_distribution'] = severity_dist.to_dict()
            
            severe_count = sum(severity_dist.get(sev, 0) for sev in ['Severe', 'Grade 3', 'Grade 4', 'Life-threatening'])
            total_ae = len(data)
            severe_pct = severe_count / total_ae * 100
            
            insights.append(f"Total adverse events: {total_ae}")
            insights.append(f"Severe adverse events: {severe_count} ({severe_pct:.1f}%)")
            
            if severe_pct > 5:
                insights.append("High rate of severe adverse events detected")
                recommendations.append("Consider immediate safety review")
                recommendations.append("Evaluate dose modifications or treatment discontinuation")
        
        return insights, recommendations, results
    
    def _analyze_laboratory_safety(self, data: pd.DataFrame) -> tuple:
        """Analyze laboratory safety data"""
        insights = []
        recommendations = []
        results = {}
        
        # Look for lab-related columns
        lab_columns = [col for col in data.columns if any(keyword in col.lower() for keyword in ['lab', 'test', 'result'])]
        
        if not lab_columns:
            insights.append("No laboratory data identified")
            return insights, recommendations, results
        
        # Analyze abnormal lab values
        abnormal_count = 0
        total_lab_values = 0
        
        for col in lab_columns:
            if data[col].dtype in ['int64', 'float64']:
                total_lab_values += len(data)
                # Simple abnormal detection (values > 3 std from mean)
                mean_val = data[col].mean()
                std_val = data[col].std()
                abnormal = ((data[col] > mean_val + 3*std_val) | (data[col] < mean_val - 3*std_val)).sum()
                abnormal_count += abnormal
        
        if total_lab_values > 0:
            abnormal_pct = abnormal_count / total_lab_values * 100
            insights.append(f"Laboratory abnormal rate: {abnormal_pct:.1f}%")
            
            if abnormal_pct > 10:
                insights.append("High rate of abnormal laboratory values")
                recommendations.append("Review laboratory testing procedures")
                recommendations.append("Consider additional safety monitoring")
        
        return insights, recommendations, results
    
    def _calculate_safety_confidence(self, context: AgentContext, safety_columns: List[str]) -> float:
        """Calculate confidence score for safety analysis"""
        base_confidence = 0.7  # Start with good confidence for safety analysis
        
        # Adjust for data completeness
        completeness = (1 - context.data.isnull().sum().sum() / (context.data.shape[0] * context.data.shape[1]))
        base_confidence *= completeness
        
        # Adjust for safety data coverage
        coverage_factor = len(safety_columns) / 3  # Max 3 safety domains
        base_confidence *= min(1.0, coverage_factor)
        
        return min(1.0, max(0.0, base_confidence))
