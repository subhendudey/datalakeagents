"""
Clinical Data Analyst Agent

Specialized agent for analyzing clinical trial data with deep
understanding of clinical domains, CDISC standards, and medical terminology.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from loguru import logger

from .base_agent import BaseAgent, AgentCapability, AgentContext, AgentResult
from ..meta_layer.semantic_model import SemanticModel, ClinicalDomain


class ClinicalDataAnalyst(BaseAgent):
    """Agent specialized in clinical data analysis"""
    
    def __init__(self, llm=None):
        super().__init__(
            name="Clinical Data Analyst",
            llm=llm,
            capabilities=[
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.SEMANTIC_UNDERSTANDING,
                AgentCapability.HYPOTHESIS_GENERATION,
                AgentCapability.REPORT_GENERATION
            ]
        )
        
        # Clinical analysis patterns
        self.analysis_patterns = {
            "demographics": self._analyze_demographics,
            "vital_signs": self._analyze_vital_signs,
            "adverse_events": self._analyze_adverse_events,
            "laboratory": self._analyze_laboratory_data,
            "efficacy": self._analyze_efficacy_endpoints
        }
    
    def can_handle(self, context: AgentContext) -> bool:
        """Check if agent can handle the context"""
        # Check if data contains clinical indicators
        clinical_columns = self._identify_clinical_columns(context.data)
        
        if len(clinical_columns) == 0:
            return False
        
        # Check if semantic model is available
        if context.semantic_model is None:
            logger.warning("No semantic model provided for clinical analysis")
        
        return True
    
    def execute(self, context: AgentContext) -> AgentResult:
        """Execute clinical data analysis"""
        start_time = datetime.now()
        
        try:
            if not self.validate_context(context):
                return AgentResult(
                    agent_name=self.name,
                    capability=AgentCapability.DATA_ANALYSIS,
                    success=False,
                    error_message="Invalid context provided"
                )
            
            # Identify clinical domains in data
            domains = self._identify_clinical_domains(context.data, context.semantic_model)
            
            # Perform domain-specific analyses
            insights = []
            recommendations = []
            result_data = {}
            
            for domain in domains:
                if domain in self.analysis_patterns:
                    domain_insights, domain_recommendations, domain_results = self.analysis_patterns[domain](context)
                    insights.extend(domain_insights)
                    recommendations.extend(domain_recommendations)
                    result_data[domain] = domain_results
            
            # Generate overall clinical insights
            overall_insights = self._generate_overall_insights(context, domains)
            insights.extend(overall_insights)
            
            # Calculate confidence score
            confidence = self._calculate_clinical_confidence(context, domains)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = AgentResult(
                agent_name=self.name,
                capability=AgentCapability.DATA_ANALYSIS,
                success=True,
                result_data=result_data,
                insights=insights,
                recommendations=recommendations,
                confidence_score=confidence,
                execution_time=execution_time,
                metadata={
                    "analyzed_domains": domains,
                    "clinical_columns": self._identify_clinical_columns(context.data),
                    "sample_size": len(context.data)
                }
            )
            
            self.log_execution(result)
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in clinical analysis: {e}")
            
            return AgentResult(
                agent_name=self.name,
                capability=AgentCapability.DATA_ANALYSIS,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _identify_clinical_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify clinical columns in the dataset"""
        clinical_keywords = [
            'patient', 'subject', 'age', 'gender', 'sex', 'weight', 'height',
            'blood_pressure', 'bp', 'systolic', 'diastolic', 'heart_rate', 'hr',
            'temperature', 'temp', 'adverse', 'ae', 'event', 'lab', 'laboratory',
            'treatment', 'drug', 'medication', 'dose', 'efficacy', 'endpoint',
            'response', 'outcome', 'survival', 'progression', 'remission'
        ]
        
        clinical_columns = []
        for column in data.columns:
            column_lower = column.lower()
            if any(keyword in column_lower for keyword in clinical_keywords):
                clinical_columns.append(column)
        
        return clinical_columns
    
    def _identify_clinical_domains(self, data: pd.DataFrame, semantic_model: Optional[SemanticModel]) -> List[str]:
        """Identify clinical domains present in the data"""
        domains = []
        columns = data.columns.tolist()
        
        # Demographics domain
        if any(col.lower() in ['age', 'gender', 'sex', 'weight', 'height'] for col in columns):
            domains.append("demographics")
        
        # Vital signs domain
        if any(keyword in ' '.join(columns).lower() for keyword in ['blood_pressure', 'bp', 'heart_rate', 'temperature']):
            domains.append("vital_signs")
        
        # Adverse events domain
        if any(keyword in ' '.join(columns).lower() for keyword in ['adverse', 'ae', 'event']):
            domains.append("adverse_events")
        
        # Laboratory domain
        if any(keyword in ' '.join(columns).lower() for keyword in ['lab', 'laboratory', 'test', 'result']):
            domains.append("laboratory")
        
        # Efficacy domain
        if any(keyword in ' '.join(columns).lower() for keyword in ['efficacy', 'endpoint', 'response', 'outcome']):
            domains.append("efficacy")
        
        # Use semantic model if available
        if semantic_model:
            for domain in ClinicalDomain:
                domain_concepts = semantic_model.get_concepts_by_domain(domain)
                concept_columns = [concept.concept_id for concept in domain_concepts]
                if any(col in concept_columns for col in columns):
                    if domain.value not in domains:
                        domains.append(domain.value)
        
        return domains
    
    def _analyze_demographics(self, context: AgentContext) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Analyze demographics data"""
        data = context.data
        insights = []
        recommendations = []
        results = {}
        
        # Age analysis
        if 'age' in data.columns:
            age_stats = {
                'mean': data['age'].mean(),
                'std': data['age'].std(),
                'min': data['age'].min(),
                'max': data['age'].max(),
                'median': data['age'].median()
            }
            results['age_statistics'] = age_stats
            
            insights.append(f"Mean patient age: {age_stats['mean']:.1f} years (±{age_stats['std']:.1f})")
            
            # Age distribution concerns
            if age_stats['min'] < 18:
                insights.append("Study includes pediatric patients (<18 years)")
                recommendations.append("Consider age-stratified analysis for pediatric subgroup")
            
            if age_stats['max'] > 85:
                insights.append("Study includes elderly patients (>85 years)")
                recommendations.append("Monitor for age-related adverse events")
        
        # Gender analysis
        if 'gender' in data.columns or 'sex' in data.columns:
            gender_col = 'gender' if 'gender' in data.columns else 'sex'
            gender_dist = data[gender_col].value_counts()
            results['gender_distribution'] = gender_dist.to_dict()
            
            # Check gender balance
            total_patients = len(data)
            male_pct = gender_dist.get('Male', gender_dist.get('M', 0)) / total_patients * 100
            female_pct = gender_dist.get('Female', gender_dist.get('F', 0)) / total_patients * 100
            
            insights.append(f"Gender distribution: {male_pct:.1f}% male, {female_pct:.1f}% female")
            
            if abs(male_pct - female_pct) > 20:
                recommendations.append("Consider gender imbalance in statistical analysis")
        
        return insights, recommendations, results
    
    def _analyze_vital_signs(self, context: AgentContext) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Analyze vital signs data"""
        data = context.data
        insights = []
        recommendations = []
        results = {}
        
        # Blood pressure analysis
        bp_columns = [col for col in data.columns if 'blood_pressure' in col.lower() or 'bp' in col.lower()]
        
        for bp_col in bp_columns:
            if data[bp_col].dtype in ['int64', 'float64']:
                bp_stats = {
                    'mean': data[bp_col].mean(),
                    'std': data[bp_col].std(),
                    'abnormal_count': sum((data[bp_col] > 140) | (data[bp_col] < 90))
                }
                results[f'{bp_col}_statistics'] = bp_stats
                
                abnormal_pct = bp_stats['abnormal_count'] / len(data) * 100
                insights.append(f"{bp_col}: {bp_stats['mean']:.1f} ± {bp_stats['std']:.1f}")
                
                if abnormal_pct > 10:
                    insights.append(f"{abnormal_pct:.1f}% of patients have abnormal {bp_col}")
                    recommendations.append(f"Monitor patients with abnormal {bp_col} closely")
        
        # Heart rate analysis
        hr_columns = [col for col in data.columns if 'heart_rate' in col.lower() or 'hr' in col.lower()]
        
        for hr_col in hr_columns:
            if data[hr_col].dtype in ['int64', 'float64']:
                hr_stats = {
                    'mean': data[hr_col].mean(),
                    'abnormal_high': sum(data[hr_col] > 100),
                    'abnormal_low': sum(data[hr_col] < 60)
                }
                results[f'{hr_col}_statistics'] = hr_stats
                
                insights.append(f"Mean {hr_col}: {hr_stats['mean']:.1f} bpm")
                
                total_abnormal = hr_stats['abnormal_high'] + hr_stats['abnormal_low']
                if total_abnormal > len(data) * 0.1:
                    recommendations.append(f"Investigate abnormal {hr_col} values")
        
        return insights, recommendations, results
    
    def _analyze_adverse_events(self, context: AgentContext) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Analyze adverse events data"""
        data = context.data
        insights = []
        recommendations = []
        results = {}
        
        # Look for adverse event columns
        ae_columns = [col for col in data.columns if 'adverse' in col.lower() or 'ae' in col.lower()]
        
        if not ae_columns:
            insights.append("No adverse event data identified")
            return insights, recommendations, results
        
        # Analyze adverse event severity
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
                recommendations.append("High rate of severe adverse events requires immediate attention")
                recommendations.append("Consider dose modifications or treatment discontinuation")
        
        return insights, recommendations, results
    
    def _analyze_laboratory_data(self, context: AgentContext) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Analyze laboratory data"""
        data = context.data
        insights = []
        recommendations = []
        results = {}
        
        # Identify laboratory columns
        lab_keywords = ['hemoglobin', 'wbc', 'rbc', 'platelet', 'creatinine', 'alt', 'ast', 'bilirubin']
        lab_columns = []
        
        for col in data.columns:
            if any(keyword in col.lower() for keyword in lab_keywords):
                lab_columns.append(col)
        
        for lab_col in lab_columns:
            if data[lab_col].dtype in ['int64', 'float64']:
                lab_stats = {
                    'mean': data[lab_col].mean(),
                    'std': data[lab_col].std(),
                    'abnormal_high': 0,
                    'abnormal_low': 0
                }
                
                # Define normal ranges (simplified)
                normal_ranges = {
                    'hemoglobin': (12, 16),
                    'wbc': (4000, 11000),
                    'platelet': (150000, 450000),
                    'creatinine': (0.6, 1.3)
                }
                
                for test, (low, high) in normal_ranges.items():
                    if test in lab_col.lower():
                        lab_stats['abnormal_high'] = sum(data[lab_col] > high)
                        lab_stats['abnormal_low'] = sum(data[lab_col] < low)
                        break
                
                results[f'{lab_col}_statistics'] = lab_stats
                
                total_abnormal = lab_stats['abnormal_high'] + lab_stats['abnormal_low']
                if total_abnormal > 0:
                    abnormal_pct = total_abnormal / len(data) * 100
                    insights.append(f"{lab_col}: {abnormal_pct:.1f}% abnormal values")
                    
                    if abnormal_pct > 20:
                        recommendations.append(f"High abnormal rate in {lab_col} - review lab procedures")
        
        return insights, recommendations, results
    
    def _analyze_efficacy_endpoints(self, context: AgentContext) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Analyze efficacy endpoints"""
        data = context.data
        insights = []
        recommendations = []
        results = {}
        
        # Identify efficacy columns
        efficacy_keywords = ['response', 'remission', 'progression', 'survival', 'endpoint']
        efficacy_columns = []
        
        for col in data.columns:
            if any(keyword in col.lower() for keyword in efficacy_keywords):
                efficacy_columns.append(col)
        
        for eff_col in efficacy_columns:
            if data[eff_col].dtype == 'object':
                # Categorical efficacy data
                response_dist = data[eff_col].value_counts()
                results[f'{eff_col}_distribution'] = response_dist.to_dict()
                
                # Calculate response rate if applicable
                positive_responses = sum(response_dist.get(resp, 0) for resp in ['Complete Response', 'Partial Response', 'Responder'])
                total_patients = len(data)
                
                if positive_responses > 0:
                    response_rate = positive_responses / total_patients * 100
                    insights.append(f"{eff_col}: {response_rate:.1f}% overall response rate")
                    
                    if response_rate < 20:
                        recommendations.append("Low response rate - consider alternative treatment strategies")
                    elif response_rate > 80:
                        recommendations.append("High response rate - validate data quality")
        
        return insights, recommendations, results
    
    def _generate_overall_insights(self, context: AgentContext, domains: List[str]) -> List[str]:
        """Generate overall clinical insights"""
        insights = []
        
        # Sample size assessment
        sample_size = len(context.data)
        if sample_size < 50:
            insights.append("Small sample size - results may not be statistically robust")
        elif sample_size > 1000:
            insights.append("Large sample size - results likely to be statistically significant")
        
        # Data completeness
        completeness = (1 - context.data.isnull().sum().sum() / (context.data.shape[0] * context.data.shape[1])) * 100
        insights.append(f"Overall data completeness: {completeness:.1f}%")
        
        if completeness < 90:
            insights.append("Significant missing data - consider imputation strategies")
        
        # Domain coverage
        insights.append(f"Clinical domains analyzed: {', '.join(domains)}")
        
        return insights
    
    def _calculate_clinical_confidence(self, context: AgentContext, domains: List[str]) -> float:
        """Calculate confidence score for clinical analysis"""
        base_confidence = 0.8  # Start with high confidence for clinical analysis
        
        # Adjust for sample size
        sample_size = len(context.data)
        if sample_size < 30:
            base_confidence -= 0.3
        elif sample_size < 100:
            base_confidence -= 0.1
        
        # Adjust for data completeness
        completeness = (1 - context.data.isnull().sum().sum() / (context.data.shape[0] * context.data.shape[1]))
        base_confidence *= completeness
        
        # Adjust for domain coverage
        if len(domains) >= 3:
            base_confidence += 0.1
        
        # Adjust for semantic model availability
        if context.semantic_model:
            base_confidence += 0.1
        
        return min(1.0, max(0.0, base_confidence))
