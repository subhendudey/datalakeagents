"""
Statistical Analysis Agent

Specialized agent for statistical analysis of clinical trial data
including hypothesis testing and statistical significance.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from .base_agent import BaseAgent, AgentCapability, AgentContext, AgentResult


class StatisticalAnalysisAgent(BaseAgent):
    """Agent specialized in statistical analysis"""
    
    def __init__(self, llm=None):
        super().__init__(
            name="Statistical Analyzer",
            llm=llm,
            capabilities=[
                AgentCapability.STATISTICAL_ANALYSIS,
                AgentCapability.HYPOTHESIS_GENERATION,
                AgentCapability.DATA_ANALYSIS
            ]
        )
    
    def can_handle(self, context: AgentContext) -> bool:
        """Check if agent can handle the context"""
        # Check if data has numeric columns for statistical analysis
        numeric_columns = context.data.select_dtypes(include=[np.number]).columns
        return len(numeric_columns) > 0 and len(context.data) > 10
    
    def execute(self, context: AgentContext) -> AgentResult:
        """Execute statistical analysis"""
        start_time = datetime.now()
        
        try:
            if not self.validate_context(context):
                return AgentResult(
                    agent_name=self.name,
                    capability=AgentCapability.STATISTICAL_ANALYSIS,
                    success=False,
                    error_message="Invalid context provided"
                )
            
            numeric_columns = context.data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return AgentResult(
                    agent_name=self.name,
                    capability=AgentCapability.STATISTICAL_ANALYSIS,
                    success=False,
                    error_message="No numeric data available for statistical analysis"
                )
            
            # Perform statistical analysis
            insights = []
            recommendations = []
            result_data = {}
            
            # Descriptive statistics
            desc_stats = self._descriptive_statistics(context.data, numeric_columns)
            result_data['descriptive_statistics'] = desc_stats
            
            # Correlation analysis
            if len(numeric_columns) > 1:
                corr_analysis = self._correlation_analysis(context.data, numeric_columns)
                result_data['correlation_analysis'] = corr_analysis
                
                # Add insights from correlation
                high_corr = corr_analysis.get('high_correlations', [])
                if high_corr:
                    insights.append(f"Found {len(high_corr)} highly correlated variable pairs")
                    for var1, var2, corr in high_corr[:3]:  # Top 3
                        insights.append(f"Strong correlation between {var1} and {var2}: {corr:.3f}")
            
            # Distribution analysis
            dist_analysis = self._distribution_analysis(context.data, numeric_columns)
            result_data['distribution_analysis'] = dist_analysis
            
            # Outlier detection
            outlier_analysis = self._outlier_detection(context.data, numeric_columns)
            result_data['outlier_analysis'] = outlier_analysis
            
            # Add insights from outlier analysis
            total_outliers = sum(outlier_analysis.values())
            if total_outliers > 0:
                insights.append(f"Detected {total_outliers} outliers across {len(numeric_columns)} variables")
                recommendations.append("Review outliers for data quality issues")
            
            # Sample size assessment
            sample_size_insights = self._sample_size_assessment(context.data)
            insights.extend(sample_size_insights)
            
            # Calculate confidence score
            confidence = self._calculate_statistical_confidence(context, numeric_columns)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = AgentResult(
                agent_name=self.name,
                capability=AgentCapability.STATISTICAL_ANALYSIS,
                success=True,
                result_data=result_data,
                insights=insights,
                recommendations=recommendations,
                confidence_score=confidence,
                execution_time=execution_time,
                metadata={
                    "numeric_columns": list(numeric_columns),
                    "sample_size": len(context.data)
                }
            )
            
            self.log_execution(result)
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in statistical analysis: {e}")
            
            return AgentResult(
                agent_name=self.name,
                capability=AgentCapability.STATISTICAL_ANALYSIS,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _descriptive_statistics(self, data: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
        """Calculate descriptive statistics"""
        stats = {}
        
        for col in numeric_columns:
            col_stats = {
                'count': data[col].count(),
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'q25': data[col].quantile(0.25),
                'q75': data[col].quantile(0.75),
                'missing_count': data[col].isnull().sum(),
                'missing_percentage': (data[col].isnull().sum() / len(data)) * 100
            }
            stats[col] = col_stats
        
        return stats
    
    def _correlation_analysis(self, data: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
        """Perform correlation analysis"""
        correlation_matrix = data[numeric_columns].corr()
        
        # Find high correlations (> 0.7 or < -0.7)
        high_correlations = []
        for i in range(len(numeric_columns)):
            for j in range(i+1, len(numeric_columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    high_correlations.append((numeric_columns[i], numeric_columns[j], corr_value))
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': high_correlations
        }
    
    def _distribution_analysis(self, data: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
        """Analyze distributions"""
        distributions = {}
        
        for col in numeric_columns:
            col_data = data[col].dropna()
            
            # Basic distribution metrics
            skewness = col_data.skew()
            kurtosis = col_data.kurtosis()
            
            # Normality assessment (simple heuristic)
            normal_indicators = abs(skewness) < 2 and abs(kurtosis) < 7
            
            distributions[col] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'appears_normal': normal_indicators,
                'distribution_type': 'approximately_normal' if normal_indicators else 'non_normal'
            }
        
        return distributions
    
    def _outlier_detection(self, data: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, int]:
        """Detect outliers using IQR method"""
        outlier_counts = {}
        
        for col in numeric_columns:
            col_data = data[col].dropna()
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            outlier_counts[col] = len(outliers)
        
        return outlier_counts
    
    def _sample_size_assessment(self, data: pd.DataFrame) -> List[str]:
        """Assess sample size adequacy"""
        insights = []
        sample_size = len(data)
        
        if sample_size < 30:
            insights.append("Small sample size (<30) - statistical power may be limited")
        elif sample_size < 100:
            insights.append("Moderate sample size - consider limitations for subgroup analyses")
        elif sample_size > 1000:
            insights.append("Large sample size - statistical significance likely even for small effects")
        else:
            insights.append("Adequate sample size for most statistical analyses")
        
        return insights
    
    def _calculate_statistical_confidence(self, context: AgentContext, numeric_columns: List[str]) -> float:
        """Calculate confidence score for statistical analysis"""
        base_confidence = 0.8  # Start with good confidence
        
        # Adjust for sample size
        sample_size = len(context.data)
        if sample_size < 30:
            base_confidence -= 0.3
        elif sample_size < 100:
            base_confidence -= 0.1
        
        # Adjust for data completeness
        completeness = (1 - context.data.isnull().sum().sum() / (context.data.shape[0] * context.data.shape[1]))
        base_confidence *= completeness
        
        # Adjust for number of variables
        if len(numeric_columns) > 10:
            base_confidence -= 0.1  # Multiple comparisons concern
        
        return min(1.0, max(0.0, base_confidence))
