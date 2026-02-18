# Statistical Analysis Agent

## 🎯 Overview

The **Statistical Analysis Agent** is a specialized AI agent designed for comprehensive statistical analysis of clinical trial data, including hypothesis testing, power analysis, and advanced statistical modeling.

## 📊 Capabilities

### Core Capabilities
- **STATISTICAL_ANALYSIS**: Comprehensive statistical testing and modeling
- **HYPOTHESIS_TESTING**: Automated hypothesis generation and testing
- **POWER_ANALYSIS**: Sample size and power calculations
- **MODELING**: Advanced statistical modeling and simulation

### Statistical Areas Covered
- **Descriptive Statistics**: Summary statistics and data characterization
- **Inferential Statistics**: Hypothesis testing and confidence intervals
- **Regression Analysis**: Linear, logistic, and Cox regression models
- **Survival Analysis**: Time-to-event analysis and survival curves
- **Bayesian Analysis**: Bayesian modeling and inference
- **Multivariate Analysis**: Principal component analysis, clustering

## 🚀 Usage

### CLI Usage
```bash
# Statistical analysis of single dataset
python src/cli.py analyze "dataset_id" --agents "statistical_analyzer"

# Batch statistical analysis across all datasets
python src/cli.py analyze-all --layer processed --agents "statistical_analyzer"
```

### Python API Usage
```python
from src.agents.statistical_analyzer import StatisticalAnalysisAgent
from src.agents.base_agent import AgentContext

# Initialize the agent
stat_analyzer = StatisticalAnalysisAgent()

# Create context
context = AgentContext(
    dataset_id="efficacy_data_001",
    data=efficacy_data,
    metadata={"domain": "efficacy"},
    semantic_model=semantic_model,
    user_query="Perform comprehensive statistical analysis of treatment efficacy"
)

# Execute statistical analysis
result = stat_analyzer.execute(context)

# View results
print(f"Statistical Analysis: {result.success}")
print(f"P-values: {result.p_values}")
print(f"Effect Sizes: {result.effect_sizes}")
print(f"Model Results: {result.models}")
```

## 📊 Statistical Analysis Features

### Descriptive Statistics
- **Central Tendency**: Mean, median, mode, geometric mean
- **Dispersion**: Standard deviation, variance, range, IQR
- **Distribution Analysis**: Skewness, kurtosis, normality tests
- **Correlation Analysis**: Pearson, Spearman, Kendall correlations
- **Missing Data Analysis**: Missing data patterns and impact

### Inferential Statistics
- **t-Tests**: One-sample, two-sample, paired t-tests
- **ANOVA**: One-way, two-way, repeated measures ANOVA
- **Non-parametric Tests**: Wilcoxon, Mann-Whitney, Kruskal-Wallis
- **Chi-Square Tests**: Independence, goodness-of-fit tests
- **Exact Tests**: Fisher's exact test, permutation tests

### Regression Analysis
- **Linear Regression**: Simple and multiple linear regression
- **Logistic Regression**: Binary and multinomial logistic regression
- **Cox Regression**: Survival analysis and hazard ratios
- **Mixed Effects Models**: Hierarchical and longitudinal models
- **Generalized Linear Models**: GLM with various distributions

### Survival Analysis
- **Kaplan-Meier Estimation**: Survival curve estimation
- **Log-Rank Tests**: Survival curve comparisons
- **Cox Proportional Hazards**: Hazard ratio estimation
- **Parametric Survival Models**: Weibull, exponential models
- **Competing Risks**: Multiple event types analysis

### Bayesian Analysis
- **Bayesian Inference**: Posterior distribution estimation
- **MCMC Methods**: Markov Chain Monte Carlo sampling
- **Bayesian Hierarchical Models**: Multi-level Bayesian models
- **Bayesian Model Comparison**: Bayes factors and model selection
- **Predictive Inference**: Bayesian prediction intervals

### Multivariate Analysis
- **Principal Component Analysis**: Dimensionality reduction
- **Factor Analysis**: Latent variable identification
- **Cluster Analysis**: K-means, hierarchical clustering
- **Discriminant Analysis**: Classification and discrimination
- **Canonical Correlation**: Multivariate relationships

## 🔍 Output Examples

### Statistical Analysis Results
```python
{
    "success": True,
    "confidence_score": 0.94,
    "execution_time": 4.56,
    "descriptive_stats": {
        "treatment_group": {
            "n": 250,
            "mean": 45.2,
            "std": 12.3,
            "median": 44.8,
            "iqr": [35.1, 55.6],
            "normality_test": {"statistic": 0.98, "p_value": 0.12}
        },
        "control_group": {
            "n": 245,
            "mean": 38.7,
            "std": 11.8,
            "median": 38.2,
            "iqr": [29.4, 48.1],
            "normality_test": {"statistic": 0.97, "p_value": 0.08}
        }
    },
    "hypothesis_tests": [
        {
            "test": "Two-sample t-test",
            "null_hypothesis": "No difference in means between groups",
            "statistic": 5.23,
            "p_value": 0.0001,
            "effect_size": 0.52,
            "confidence_interval": [4.8, 12.2],
            "conclusion": "Reject null hypothesis - significant difference detected"
        }
    ],
    "models": [
        {
            "type": "Linear Regression",
            "dependent_variable": "efficacy_score",
            "independent_variables": ["treatment", "age", "baseline_score"],
            "r_squared": 0.67,
            "adjusted_r_squared": 0.65,
            "f_statistic": 45.6,
            "coefficients": {
                "treatment": {"estimate": 6.5, "std_error": 1.2, "p_value": 0.0001},
                "age": {"estimate": -0.1, "std_error": 0.05, "p_value": 0.04},
                "baseline_score": {"estimate": 0.8, "std_error": 0.1, "p_value": 0.0001}
            }
        }
    ],
    "power_analysis": {
        "achieved_power": 0.89,
        "required_sample_size": 180,
        "actual_sample_size": 495,
        "effect_size_detected": 0.52,
        "alpha": 0.05
    }
}
```

### Statistical Reports
- **Summary Statistics**: Comprehensive descriptive statistics
- **Hypothesis Test Results**: All statistical tests with p-values
- **Model Diagnostics**: Model fit and validation statistics
- **Power Calculations**: Sample size and power analysis
- **Effect Size Estimates**: Clinical significance measures

## ⚙️ Configuration

### Statistical Parameters
```python
class StatisticalAnalysisAgent:
    def __init__(self, llm=None):
        super().__init__(
            name="Statistical Analysis Agent",
            llm=llm,
            capabilities=[
                AgentCapability.STATISTICAL_ANALYSIS,
                AgentCapability.HYPOTHESIS_TESTING,
                AgentCapability.POWER_ANALYSIS,
                AgentCapability.MODELING
            ]
        )
```

### Analysis Settings
- **Significance Level**: α = 0.05 (default)
- **Power Target**: 80% (default)
- **Effect Size Threshold**: Cohen's d = 0.5 (medium)
- **Multiple Testing Correction**: Bonferroni, FDR options
- **Missing Data Handling**: Listwise, pairwise, imputation

## 🔧 Technical Details

### Statistical Libraries
- **SciPy**: Core statistical functions and tests
- **Statsmodels**: Advanced statistical modeling
- **NumPy**: Numerical computations and arrays
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Statistical visualization

### Statistical Methods
- **Parametric Tests**: t-tests, ANOVA, regression
- **Non-parametric Tests**: Wilcoxon, Mann-Whitney, Kruskal-Wallis
- **Robust Methods**: Trimmed means, bootstrapping
- **Bayesian Methods**: MCMC, posterior inference
- **Machine Learning**: Cross-validation, model selection

### Data Requirements
- **Numeric Variables**: Continuous measurements
- **Categorical Variables**: Factor variables with proper coding
- **Time Variables**: Date/time formats for survival analysis
- **Missing Data**: Documented missing data patterns
- **Sample Size**: Adequate power for planned analyses

## 📈 Analysis Types

### Efficacy Analysis
- **Primary Endpoint Analysis**: Main efficacy hypothesis testing
- **Secondary Endpoints**: Additional efficacy measures
- **Subgroup Analysis**: Pre-specified subgroup comparisons
- **Interim Analysis**: Early efficacy assessment
- **Per-Protocol vs ITT**: Different analysis populations

### Safety Analysis
- **Adverse Event Rates**: Comparison between groups
- **Laboratory Changes**: Baseline to post-baseline changes
- **Vital Sign Changes**: Statistical significance of changes
- **Dose-Response Analysis**: Relationship between dose and outcomes
- **Time-to-Event Analysis**: Safety event timing

### Demographic Analysis
- **Baseline Comparisons**: Group balance assessment
- **Covariate Adjustment**: Confounding variable control
- **Stratified Analysis**: Analysis within strata
- **Predictive Factors**: Baseline predictors of outcome
- **Prognostic Models**: Risk factor identification

## 🚨 Statistical Considerations

### Assumptions Checking
- **Normality**: Shapiro-Wilk, Kolmogorov-Smirnov tests
- **Homogeneity of Variance**: Levene's test, Bartlett's test
- **Independence**: Durbin-Watson test, autocorrelation
- **Linearity**: Residual plots, component analysis
- **Multicollinearity**: VIF, condition number assessment

### Multiple Testing
- **Family-wise Error Rate**: Bonferroni correction
- **False Discovery Rate**: Benjamini-Hochberg procedure
- **Hierarchical Testing**: Pre-specified testing hierarchy
- **Group Sequential Testing**: Interim analysis adjustments
- **Adaptive Designs**: Sample size re-estimation

### Missing Data
- **Missing Completely at Random (MCAR)**: Little's test
- **Missing at Random (MAR)**: Conditional independence
- **Missing Not at Random (MNAR)**: Sensitivity analysis
- **Imputation Methods**: Multiple imputation, EM algorithm
- **Complete Case Analysis**: Listwise deletion considerations

## 📊 Visualization

### Statistical Plots
- **Histograms**: Distribution visualization
- **Box Plots**: Group comparisons and outliers
- **Scatter Plots**: Relationship visualization
- **Survival Curves**: Kaplan-Meier plots
- **Forest Plots**: Meta-analysis and subgroup results

### Diagnostic Plots
- **Q-Q Plots**: Normality assessment
- **Residual Plots**: Model diagnostics
- **Leverage Plots**: Influence diagnostics
- **ROC Curves**: Classification performance
- **Calibration Plots**: Model calibration

## 🔄 Future Enhancements

### Planned Features
- **Adaptive Designs**: Sample size re-estimation
- **Bayesian Methods**: Advanced Bayesian modeling
- **Machine Learning**: Predictive modeling integration
- **Real-time Analysis**: Streaming data analysis

### Advanced Methods
- **Causal Inference**: Propensity score methods
- **Mediation Analysis**: Indirect effect estimation
- **Longitudinal Analysis**: Mixed effects models
- **Network Meta-analysis**: Multiple treatment comparison

## 📚 References

### Statistical Standards
- **ICH E9**: Statistical Principles for Clinical Trials
- **ICH E9(R1)**: Estimands and Sensitivity Analysis
- **FDA Guidance**: Statistical analysis requirements
- **EMA Guideline**: Statistical analysis in clinical trials

### Statistical Literature
- **Fleiss**: Statistical Methods for Rates and Proportions
- **Altman**: Practical Statistics for Medical Research
- **Hosmer**: Applied Logistic Regression
- **Kleinbaum**: Survival Analysis

## 🆘 Support

### Troubleshooting
- **Convergence Issues**: Check model specifications
- **Assumption Violations**: Consider alternative methods
- **Sample Size Problems**: Power analysis recommendations
- **Computational Issues**: Optimize data processing

### Common Issues
1. **Non-normal Data**: Use non-parametric methods
2. **Small Sample Sizes**: Exact tests or Bayesian methods
3. **Missing Data**: Appropriate imputation strategies
4. **Multiple Comparisons**: Adjust for multiple testing

---

**Built with ❤️ for rigorous statistical analysis in clinical research**
