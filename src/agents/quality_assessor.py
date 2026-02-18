"""
Data Quality Assessment Agent

Specialized agent for assessing data quality in clinical trial data
including completeness, accuracy, and consistency checks.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from .base_agent import BaseAgent, AgentCapability, AgentContext, AgentResult


class DataQualityAgent(BaseAgent):
    """Agent specialized in data quality assessment"""
    
    def __init__(self, llm=None):
        super().__init__(
            name="Data Quality Assessor",
            llm=llm,
            capabilities=[
                AgentCapability.QUALITY_ASSESSMENT,
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.ANOMALY_DETECTION
            ]
        )
    
    def can_handle(self, context: AgentContext) -> bool:
        """Check if agent can handle the context"""
        # Can handle any data for quality assessment
        return True
    
    def execute(self, context: AgentContext) -> AgentResult:
        """Execute data quality assessment"""
        start_time = datetime.now()
        
        try:
            if not self.validate_context(context):
                return AgentResult(
                    agent_name=self.name,
                    capability=AgentCapability.QUALITY_ASSESSMENT,
                    success=False,
                    error_message="Invalid context provided"
                )
            
            # Perform quality assessment
            insights = []
            recommendations = []
            result_data = {}
            
            # Completeness assessment
            completeness = self._assess_completeness(context.data)
            result_data['completeness'] = completeness
            
            # Add insights from completeness
            overall_completeness = completeness['overall_completeness']
            insights.append(f"Overall data completeness: {overall_completeness:.1%}")
            
            if overall_completeness < 0.9:
                insights.append("Data completeness below 90% threshold")
                recommendations.append("Investigate missing data patterns")
                recommendations.append("Consider data imputation strategies")
            
            # Accuracy assessment
            accuracy = self._assess_accuracy(context.data)
            result_data['accuracy'] = accuracy
            
            # Consistency assessment
            consistency = self._assess_consistency(context.data)
            result_data['consistency'] = consistency
            
            # Uniqueness assessment
            uniqueness = self._assess_uniqueness(context.data)
            result_data['uniqueness'] = uniqueness
            
            # Validity assessment
            validity = self._assess_validity(context.data)
            result_data['validity'] = validity
            
            # Overall quality score
            quality_score = self._calculate_overall_quality(completeness, accuracy, consistency, uniqueness, validity)
            result_data['overall_quality_score'] = quality_score
            
            insights.append(f"Overall data quality score: {quality_score:.2f}/1.0")
            
            # Quality categorization
            if quality_score >= 0.9:
                insights.append("Data quality is excellent")
            elif quality_score >= 0.8:
                insights.append("Data quality is good")
            elif quality_score >= 0.6:
                insights.append("Data quality is fair - improvement needed")
                recommendations.append("Focus on improving data collection processes")
            else:
                insights.append("Data quality is poor - significant issues found")
                recommendations.append("Major data quality remediation required")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = AgentResult(
                agent_name=self.name,
                capability=AgentCapability.QUALITY_ASSESSMENT,
                success=True,
                result_data=result_data,
                insights=insights,
                recommendations=recommendations,
                confidence_score=quality_score,
                execution_time=execution_time,
                metadata={
                    "sample_size": len(context.data),
                    "column_count": len(context.data.columns)
                }
            )
            
            self.log_execution(result)
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in quality assessment: {e}")
            
            return AgentResult(
                agent_name=self.name,
                capability=AgentCapability.QUALITY_ASSESSMENT,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _assess_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data completeness"""
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        overall_completeness = 1 - (missing_cells / total_cells)
        
        # Column-level completeness
        column_completeness = {}
        for col in data.columns:
            non_null_count = data[col].count()
            completeness = non_null_count / len(data)
            column_completeness[col] = {
                'completeness': completeness,
                'missing_count': len(data) - non_null_count,
                'missing_percentage': (1 - completeness) * 100
            }
        
        # Identify high-missing columns
        high_missing_columns = [col for col, stats in column_completeness.items() if stats['completeness'] < 0.7]
        
        return {
            'overall_completeness': overall_completeness,
            'column_completeness': column_completeness,
            'high_missing_columns': high_missing_columns,
            'total_missing_cells': missing_cells,
            'total_cells': total_cells
        }
    
    def _assess_accuracy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data accuracy (basic checks)"""
        accuracy_issues = []
        
        # Check for impossible values in common clinical variables
        for col in data.columns:
            if 'age' in col.lower() and data[col].dtype in ['int64', 'float64']:
                # Age should be reasonable
                invalid_ages = data[(data[col] < 0) | (data[col] > 150)]
                if len(invalid_ages) > 0:
                    accuracy_issues.append(f"Invalid age values in {col}: {len(invalid_ages)} records")
            
            elif 'weight' in col.lower() and data[col].dtype in ['int64', 'float64']:
                # Weight should be positive
                invalid_weights = data[data[col] <= 0]
                if len(invalid_weights) > 0:
                    accuracy_issues.append(f"Invalid weight values in {col}: {len(invalid_weights)} records")
            
            elif 'height' in col.lower() and data[col].dtype in ['int64', 'float64']:
                # Height should be positive and reasonable
                invalid_heights = data[(data[col] <= 0) | (data[col] > 300)]
                if len(invalid_heights) > 0:
                    accuracy_issues.append(f"Invalid height values in {col}: {len(invalid_heights)} records")
        
        accuracy_score = 1.0 - (len(accuracy_issues) / data.shape[1])  # Simple scoring
        
        return {
            'accuracy_score': max(0.0, accuracy_score),
            'accuracy_issues': accuracy_issues,
            'issue_count': len(accuracy_issues)
        }
    
    def _assess_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency"""
        consistency_issues = []
        
        # Check for duplicate rows
        duplicate_rows = data.duplicated().sum()
        if duplicate_rows > 0:
            consistency_issues.append(f"Found {duplicate_rows} duplicate rows")
        
        # Check for inconsistent categorical values
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            unique_values = data[col].dropna().unique()
            # Check for similar values that might be inconsistent (e.g., "Male" vs "M")
            if len(unique_values) > 1:
                # Simple check for case inconsistencies
                lower_values = [str(val).lower() for val in unique_values]
                if len(set(lower_values)) < len(unique_values):
                    consistency_issues.append(f"Case inconsistencies found in {col}")
        
        consistency_score = 1.0 - (len(consistency_issues) / 5)  # Simple scoring
        
        return {
            'consistency_score': max(0.0, consistency_score),
            'consistency_issues': consistency_issues,
            'duplicate_rows': duplicate_rows
        }
    
    def _assess_uniqueness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data uniqueness"""
        uniqueness_scores = {}
        
        # Check uniqueness of potential ID columns
        id_columns = [col for col in data.columns if 'id' in col.lower()]
        
        for col in id_columns:
            if col in data.columns:
                unique_ratio = data[col].nunique() / len(data)
                uniqueness_scores[col] = unique_ratio
                
                if unique_ratio < 0.9:
                    logger.warning(f"Column {col} may not be a proper unique identifier")
        
        # Overall uniqueness score
        avg_uniqueness = np.mean(list(uniqueness_scores.values())) if uniqueness_scores else 1.0
        
        return {
            'overall_uniqueness': avg_uniqueness,
            'column_uniqueness': uniqueness_scores,
            'id_columns_found': id_columns
        }
    
    def _assess_validity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data validity (format and range checks)"""
        validity_issues = []
        
        # Check date columns
        for col in data.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    dates = pd.to_datetime(data[col], errors='coerce')
                    invalid_dates = dates.isnull().sum() - data[col].isnull().sum()  # Extra nulls from invalid dates
                    if invalid_dates > 0:
                        validity_issues.append(f"Invalid date formats in {col}: {invalid_dates} records")
                except:
                    validity_issues.append(f"Could not parse dates in {col}")
        
        validity_score = 1.0 - (len(validity_issues) / data.shape[1])  # Simple scoring
        
        return {
            'validity_score': max(0.0, validity_score),
            'validity_issues': validity_issues,
            'issue_count': len(validity_issues)
        }
    
    def _calculate_overall_quality(self, completeness: Dict, accuracy: Dict, 
                                 consistency: Dict, uniqueness: Dict, validity: Dict) -> float:
        """Calculate overall quality score"""
        weights = {
            'completeness': 0.3,
            'accuracy': 0.25,
            'consistency': 0.2,
            'uniqueness': 0.15,
            'validity': 0.1
        }
        
        scores = {
            'completeness': completeness['overall_completeness'],
            'accuracy': accuracy['accuracy_score'],
            'consistency': consistency['consistency_score'],
            'uniqueness': uniqueness['overall_uniqueness'],
            'validity': validity['validity_score']
        }
        
        overall_score = sum(scores[dimension] * weights[dimension] for dimension in scores)
        
        return round(overall_score, 2)
