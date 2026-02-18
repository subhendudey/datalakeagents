# Data Quality Agent

## 🎯 Overview

The **Data Quality Agent** is a specialized AI agent designed for comprehensive assessment, monitoring, and improvement of data quality in clinical trials. It evaluates completeness, consistency, validity, and overall data integrity.

## 🔍 Capabilities

### Core Capabilities
- **QUALITY_ASSESSMENT**: Comprehensive data quality evaluation
- **COMPLETENESS_ANALYSIS**: Missing data pattern analysis
- **CONSISTENCY_CHECKING**: Logical consistency validation
- **VALIDITY_ASSESSMENT**: Data range and format validation

### Quality Dimensions Covered
- **Completeness**: Missing data assessment and patterns
- **Accuracy**: Data correctness and precision
- **Consistency**: Logical and temporal consistency
- **Validity**: Conformance to rules and constraints
- **Timeliness**: Data entry and update timeliness
- **Uniqueness**: Duplicate detection and resolution

## 🚀 Usage

### CLI Usage
```bash
# Quality assessment of single dataset
python src/cli.py analyze "dataset_id" --agents "quality_assessor"

# Batch quality assessment across all datasets
python src/cli.py analyze-all --layer processed --agents "quality_assessor"
```

### Python API Usage
```python
from src.agents.quality_assessor import DataQualityAgent
from src.agents.base_agent import AgentContext

# Initialize the agent
quality_agent = DataQualityAgent()

# Create context
context = AgentContext(
    dataset_id="clinical_data_001",
    data=clinical_data,
    metadata={"domain": "demographics"},
    semantic_model=semantic_model,
    user_query="Assess data quality and identify improvement opportunities"
)

# Execute quality assessment
result = quality_agent.execute(context)

# View results
print(f"Quality Assessment: {result.success}")
print(f"Overall Score: {result.quality_score}")
print(f"Issues Found: {len(result.quality_issues)}")
print(f"Recommendations: {result.recommendations}")
```

## 📊 Data Quality Assessment Features

### Completeness Analysis
- **Missing Data Rates**: Percentage of missing values by variable
- **Missing Data Patterns**: Systematic missing data identification
- **Item Non-response**: Variable-level missing data analysis
- **Unit Non-response**: Record-level missing data assessment
- **Longitudinal Completeness**: Missing data over time points

### Accuracy Assessment
- **Data Entry Errors**: Typographical and transcription errors
- **Measurement Errors**: Inaccurate measurements and readings
- **Coding Errors**: Incorrect coding or classification
- **Calculation Errors**: Derived variable computation errors
- **Protocol Deviations**: Deviations from data collection protocols

### Consistency Checking
- **Logical Consistency**: Rule-based consistency validation
- **Temporal Consistency**: Date and time sequence validation
- **Cross-form Consistency**: Related form consistency checks
- **Visit Consistency**: Visit-specific data consistency
- **Patient Consistency**: Within-patient data consistency

### Validity Assessment
- **Range Validation**: Value range constraint checking
- **Format Validation**: Data format and pattern validation
- **Domain Validation**: Valid value domain checking
- **Reference Validation**: Reference data consistency
- **Business Rules**: Study-specific rule validation

### Uniqueness Assessment
- **Duplicate Detection**: Exact and fuzzy duplicate identification
- **Record Linkage**: Patient record matching across sources
- **Key Variable Analysis**: Unique identifier validation
- **Redundancy Analysis**: Redundant data element identification
- **Merge Quality**: Data merge and consolidation quality

## 🔍 Output Examples

### Data Quality Assessment Results
```python
{
    "success": True,
    "confidence_score": 0.96,
    "execution_time": 2.89,
    "quality_score": 0.87,
    "quality_dimensions": {
        "completeness": {
            "score": 0.92,
            "missing_rate": 0.08,
            "issues": [
                {
                    "variable": "weight_kg",
                    "missing_count": 45,
                    "missing_rate": 0.09,
                    "pattern": "Random missing",
                    "severity": "MODERATE"
                }
            ]
        },
        "accuracy": {
            "score": 0.85,
            "error_rate": 0.03,
            "issues": [
                {
                    "variable": "age",
                    "error_count": 12,
                    "error_type": "Out of range",
                    "examples": [150, 165, 178],
                    "severity": "HIGH"
                }
            ]
        },
        "consistency": {
            "score": 0.91,
            "inconsistency_rate": 0.04,
            "issues": [
                {
                    "type": "Date inconsistency",
                    "description": "Informed consent date after enrollment date",
                    "affected_records": 23,
                    "severity": "MODERATE"
                }
            ]
        },
        "validity": {
            "score": 0.94,
            "invalid_rate": 0.02,
            "issues": [
                {
                    "variable": "gender",
                    "invalid_values": ["U", "X", "Unknown"],
                    "count": 8,
                    "severity": "LOW"
                }
            ]
        }
    },
    "quality_issues": [
        {
            "id": "Q001",
            "type": "COMPLETENESS",
            "severity": "MODERATE",
            "description": "Missing weight measurements",
            "affected_variables": ["weight_kg"],
            "affected_records": 45,
            "recommendation": "Implement weight collection protocol"
        },
        {
            "id": "Q002",
            "type": "ACCURACY",
            "severity": "HIGH",
            "description": "Age values outside plausible range",
            "affected_variables": ["age"],
            "affected_records": 12,
            "recommendation": "Verify data entry and implement range checks"
        }
    ],
    "recommendations": [
        "Implement real-time validation rules during data entry",
        "Conduct targeted data cleaning for age outliers",
        "Establish weight collection procedures",
        "Add consistency checks for date fields",
        "Implement duplicate detection protocols"
    ],
    "improvement_plan": {
        "priority_actions": [
            "Fix age outliers within 1 week",
            "Implement range validation for all numeric fields"
        ],
        "medium_term": [
            "Enhance data entry validation",
            "Conduct staff training on data quality"
        ],
        "long_term": [
            "Implement automated quality monitoring",
            "Establish data quality metrics dashboard"
        ]
    }
}
```

### Quality Score Breakdown
- **0.90-1.00**: Excellent quality
- **0.80-0.89**: Good quality
- **0.70-0.79**: Fair quality
- **0.60-0.69**: Poor quality
- **<0.60**: Unacceptable quality

## ⚙️ Configuration

### Quality Thresholds
```python
class DataQualityAgent:
    def __init__(self, llm=None):
        super().__init__(
            name="Data Quality Agent",
            llm=llm,
            capabilities=[
                AgentCapability.QUALITY_ASSESSMENT,
                AgentCapability.COMPLETENESS_ANALYSIS,
                AgentCapability.CONSISTENCY_CHECKING,
                AgentCapability.VALIDITY_ASSESSMENT
            ]
        )
```

### Quality Rules
- **Completeness Threshold**: Alert if missing rate > 10%
- **Accuracy Threshold**: Alert if error rate > 5%
- **Consistency Threshold**: Alert if inconsistency rate > 3%
- **Validity Threshold**: Alert if invalid rate > 2%
- **Overall Quality Threshold**: Alert if overall score < 0.80

## 🔧 Technical Details

### Quality Metrics
- **Completeness Rate**: (1 - missing_rate) × 100%
- **Accuracy Rate**: (1 - error_rate) × 100%
- **Consistency Rate**: (1 - inconsistency_rate) × 100%
- **Validity Rate**: (1 - invalid_rate) × 100%
- **Overall Quality Score**: Weighted average of dimensions

### Detection Algorithms
- **Pattern Recognition**: Missing data pattern identification
- **Statistical Outliers**: Z-score, IQR method for outlier detection
- **Rule Engines**: Business rule validation framework
- **Fuzzy Matching**: Approximate duplicate detection
- **Temporal Analysis**: Time-based consistency checking

### Data Validation Rules
- **Range Rules**: Value range constraints
- **Format Rules**: Data format specifications
- **Dependency Rules**: Inter-field dependencies
- **Business Rules**: Study-specific validations
- **Regulatory Rules**: Compliance requirements

## 📈 Quality Monitoring

### Continuous Monitoring
- **Real-time Validation**: Immediate data entry validation
- **Batch Validation**: Scheduled quality assessments
- **Trend Analysis**: Quality metric trends over time
- **Alert Systems**: Automated quality alerts
- **Dashboard Reporting**: Visual quality metrics

### Quality Metrics Dashboard
- **Overall Quality Score**: Aggregate quality measure
- **Dimension Scores**: Individual quality dimension performance
- **Trend Charts**: Quality metric trends over time
- **Issue Tracking**: Quality issue identification and resolution
- **Site Performance**: Site-specific quality metrics

### Quality Improvement Cycle
1. **Assessment**: Comprehensive quality evaluation
2. **Identification**: Quality issue detection
3. **Prioritization**: Issue severity and impact assessment
4. **Resolution**: Quality issue remediation
5. **Monitoring**: Ongoing quality surveillance
6. **Improvement**: Continuous quality enhancement

## 🚨 Quality Issues Classification

### Severity Levels
- **CRITICAL**: Data integrity issues affecting study validity
- **HIGH**: Significant quality issues requiring immediate action
- **MODERATE**: Quality issues needing attention
- **LOW**: Minor quality issues for information

### Issue Categories
- **Data Entry Errors**: Transcription and input errors
- **Protocol Deviations**: Deviations from collection protocols
- **System Issues**: Software or system-related problems
- **Training Issues**: Staff training deficiencies
- **Process Issues**: Workflow and procedure problems

### Resolution Strategies
- **Immediate Correction**: Real-time error correction
- **Data Cleaning**: Systematic data correction
- **Process Improvement**: Workflow enhancement
- **Training Enhancement**: Staff education programs
- **System Enhancement**: Software and tool improvements

## 📊 Reporting

### Quality Reports
- **Executive Summary**: High-level quality overview
- **Detailed Analysis**: Comprehensive quality assessment
- **Trend Reports**: Quality metric trends
- **Site Reports**: Site-specific quality performance
- **Issue Reports**: Quality issue tracking and resolution

### Key Performance Indicators
- **Data Quality Index**: Overall quality measure
- **Error Rate Reduction**: Quality improvement metrics
- **Resolution Time**: Issue resolution efficiency
- **Compliance Rate**: Regulatory compliance metrics
- **User Satisfaction**: End-user quality perception

## 🔄 Future Enhancements

### Planned Features
- **Machine Learning**: Automated quality pattern recognition
- **Real-time Monitoring**: Live data quality surveillance
- **Predictive Quality**: Quality issue prediction
- **Cross-study Analysis**: Multi-study quality benchmarking

### Advanced Analytics
- **Root Cause Analysis**: Quality issue root cause identification
- **Impact Assessment**: Quality issue impact quantification
- **Risk Assessment**: Quality-related risk evaluation
- **Cost Analysis**: Quality improvement cost-benefit analysis

## 📚 References

### Quality Standards
- **CDISC Data Quality**: Clinical data interchange standards
- **GCP Guidelines**: Good Clinical Practice requirements
- **FDA Guidance**: Data quality and integrity guidance
- **EMA Guidelines**: European data quality standards

### Quality Literature
- **Data Quality Assessment**: Comprehensive assessment frameworks
- **Clinical Data Management**: Best practices and standards
- **Data Governance**: Quality management frameworks
- **Quality Improvement**: Continuous improvement methodologies

## 🆘 Support

### Troubleshooting
- **Quality Score Issues**: Verify data completeness and accuracy
- **Validation Errors**: Check validation rule configurations
- **Performance Issues**: Optimize data processing algorithms
- **Integration Problems**: Verify data source connectivity

### Common Issues
1. **Missing Data**: Implement data collection protocols
2. **Inconsistent Data**: Establish standardization procedures
3 **Invalid Data**: Implement validation rules
4. **Duplicate Data**: Implement deduplication processes

---

**Built with ❤️ for data excellence and clinical research integrity**
