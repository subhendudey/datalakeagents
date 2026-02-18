#!/usr/bin/env python3
"""
Generate Sample Clinical Trial Data

Creates realistic sample data for testing the clinical trials data lake.
Includes demographics, vital signs, adverse events, and laboratory data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def generate_demographics(n_patients: int = 100) -> pd.DataFrame:
    """Generate patient demographics data"""
    np.random.seed(42)
    
    # Generate patient IDs
    patient_ids = [f"P{str(i+1).zfill(4)}" for i in range(n_patients)]
    
    # Generate ages (normal distribution around 55)
    ages = np.random.normal(55, 15, n_patients).astype(int)
    ages = np.clip(ages, 18, 85)
    
    # Generate genders
    genders = np.random.choice(['Male', 'Female'], n_patients, p=[0.52, 0.48])
    
    # Generate weights (kg) - correlated with age and gender
    weights = []
    for i in range(n_patients):
        base_weight = 70 + np.random.normal(0, 10)
        if genders[i] == 'Male':
            base_weight += 10
        if ages[i] > 65:
            base_weight -= 5
        weights.append(max(40, base_weight + np.random.normal(0, 5)))
    
    # Generate heights (cm)
    heights = []
    for i in range(n_patients):
        base_height = 170 if genders[i] == 'Male' else 160
        heights.append(base_height + np.random.normal(0, 8))
    
    # Generate enrollment dates
    start_date = datetime(2023, 1, 1)
    enrollment_dates = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(n_patients)]
    
    # Generate study IDs
    study_ids = np.random.choice(['STUDY001', 'STUDY002', 'STUDY003'], n_patients, p=[0.4, 0.35, 0.25])
    
    demographics = pd.DataFrame({
        'patient_id': patient_ids,
        'study_id': study_ids,
        'age': ages,
        'gender': genders,
        'weight_kg': np.round(weights, 1),
        'height_cm': np.round(heights, 1),
        'enrollment_date': enrollment_dates,
        'informed_consent_date': [d - timedelta(days=random.randint(1, 7)) for d in enrollment_dates],
        'race_ethnicity': np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], 
                                         n_patients, p=[0.6, 0.15, 0.12, 0.1, 0.03])
    })
    
    return demographics


def generate_vital_signs(demographics: pd.DataFrame, n_measurements: int = 5) -> pd.DataFrame:
    """Generate vital signs measurements"""
    records = []
    
    for _, patient in demographics.iterrows():
        for measurement in range(n_measurements):
            # Generate measurement date
            base_date = patient['enrollment_date']
            measurement_date = base_date + timedelta(days=measurement * 30)
            
            # Generate vital signs with realistic variation
            # Blood pressure
            base_sbp = 120 + (patient['age'] - 50) * 0.5
            base_dbp = 80 + (patient['age'] - 50) * 0.3
            
            sbp = max(90, base_sbp + np.random.normal(0, 10))
            dbp = max(60, base_dbp + np.random.normal(0, 7))
            
            # Heart rate
            hr = 70 + np.random.normal(0, 12)
            hr = max(40, min(120, hr))
            
            # Temperature
            temp = 36.8 + np.random.normal(0, 0.5)
            temp = max(35.0, min(40.0, temp))
            
            # Respiratory rate
            rr = 16 + np.random.normal(0, 3)
            rr = max(8, min(30, rr))
            
            records.append({
                'patient_id': patient['patient_id'],
                'study_id': patient['study_id'],
                'measurement_date': measurement_date,
                'visit_number': measurement + 1,
                'systolic_bp': round(sbp, 1),
                'diastolic_bp': round(dbp, 1),
                'heart_rate': round(hr, 0),
                'temperature': round(temp, 1),
                'respiratory_rate': round(rr, 0),
                'oxygen_saturation': min(100, max(85, 98 + np.random.normal(0, 2)))
            })
    
    return pd.DataFrame(records)


def generate_adverse_events(demographics: pd.DataFrame) -> pd.DataFrame:
    """Generate adverse events data"""
    # Define common adverse events
    ae_types = [
        ('Headache', 'Mild', 0.15),
        ('Nausea', 'Mild', 0.12),
        ('Fatigue', 'Mild', 0.18),
        ('Dizziness', 'Mild', 0.08),
        ('Rash', 'Moderate', 0.05),
        ('Diarrhea', 'Moderate', 0.07),
        ('Vomiting', 'Moderate', 0.04),
        ('Elevated liver enzymes', 'Moderate', 0.03),
        ('Chest pain', 'Severe', 0.01),
        ('Severe allergic reaction', 'Severe', 0.005)
    ]
    
    records = []
    
    for _, patient in demographics.iterrows():
        # Each patient has 0-3 adverse events
        n_ae = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
        
        for i in range(n_ae):
            # Select adverse event type
            ae_type, severity, probability = random.choice(ae_types)
            
            # Generate onset date
            onset_date = patient['enrollment_date'] + timedelta(days=random.randint(1, 180))
            
            # Generate resolution date (some events are ongoing)
            if random.random() < 0.7:  # 70% resolved
                resolution_duration = random.randint(1, 30)
                resolution_date = onset_date + timedelta(days=resolution_duration)
            else:
                resolution_date = None
            
            # Causality assessment
            causality = random.choice(['Related', 'Possibly related', 'Unrelated', 'Unlikely'], 
                                    p=[0.3, 0.4, 0.2, 0.1])
            
            records.append({
                'patient_id': patient['patient_id'],
                'study_id': patient['study_id'],
                'ae_number': i + 1,
                'ae_term': ae_type,
                'ae_severity': severity,
                'ae_onset_date': onset_date,
                'ae_resolution_date': resolution_date,
                'ae_serious': severity == 'Severe',
                'ae_related_to_study_drug': causality in ['Related', 'Possibly related'],
                'ae_action_taken': random.choice(['None', 'Dose reduced', 'Drug interrupted', 'Drug discontinued']),
                'ae_outcome': 'Recovered/Resolved' if resolution_date else 'Ongoing'
            })
    
    return pd.DataFrame(records)


def generate_laboratory_data(demographics: pd.DataFrame) -> pd.DataFrame:
    """Generate laboratory test results"""
    # Define normal lab ranges
    lab_tests = {
        'Hemoglobin': {'unit': 'g/dL', 'normal_range': (12.0, 16.0), 'mean': 14.0},
        'WBC Count': {'unit': 'x10^9/L', 'normal_range': (4.0, 11.0), 'mean': 7.0},
        'Platelets': {'unit': 'x10^9/L', 'normal_range': (150, 450), 'mean': 250},
        'Creatinine': {'unit': 'mg/dL', 'normal_range': (0.6, 1.3), 'mean': 0.9},
        'ALT': {'unit': 'U/L', 'normal_range': (10, 40), 'mean': 25},
        'AST': {'unit': 'U/L', 'normal_range': (10, 35), 'mean': 22},
        'Total Bilirubin': {'unit': 'mg/dL', 'normal_range': (0.3, 1.2), 'mean': 0.7},
        'Glucose': {'unit': 'mg/dL', 'normal_range': (70, 100), 'mean': 85}
    }
    
    records = []
    
    for _, patient in demographics.iterrows():
        # Generate 3 lab visits per patient
        for visit in range(3):
            visit_date = patient['enrollment_date'] + timedelta(days=visit * 90)
            
            for test_name, test_info in lab_tests.items():
                # Generate test result with some variation
                result = test_info['mean'] + np.random.normal(0, test_info['mean'] * 0.1)
                
                # Occasionally generate abnormal values
                if random.random() < 0.1:  # 10% abnormal
                    if random.random() < 0.5:
                        result = test_info['normal_range'][0] - abs(np.random.normal(0, test_info['mean'] * 0.2))
                    else:
                        result = test_info['normal_range'][1] + abs(np.random.normal(0, test_info['mean'] * 0.2))
                
                records.append({
                    'patient_id': patient['patient_id'],
                    'study_id': patient['study_id'],
                    'test_date': visit_date,
                    'visit_number': visit + 1,
                    'test_name': test_name,
                    'test_result': round(result, 2),
                    'test_unit': test_info['unit'],
                    'normal_range_low': test_info['normal_range'][0],
                    'normal_range_high': test_info['normal_range'][1],
                    'abnormal_flag': result < test_info['normal_range'][0] or result > test_info['normal_range'][1]
                })
    
    return pd.DataFrame(records)


def generate_efficacy_data(demographics: pd.DataFrame) -> pd.DataFrame:
    """Generate efficacy endpoint data"""
    records = []
    
    for _, patient in demographics.iterrows():
        # Generate baseline and follow-up measurements
        baseline_score = random.randint(20, 80)
        
        # Treatment response (some patients respond better)
        response_factor = random.choice([0.8, 0.9, 1.0, 1.1, 1.2], p=[0.1, 0.2, 0.4, 0.2, 0.1])
        
        for timepoint in [0, 4, 8, 12, 24]:  # weeks
            measurement_date = patient['enrollment_date'] + timedelta(weeks=timepoint)
            
            if timepoint == 0:
                score = baseline_score
            else:
                # Score improves over time for responders
                improvement = (timepoint / 24) * 30 * response_factor
                score = max(0, baseline_score - improvement + np.random.normal(0, 5))
            
            records.append({
                'patient_id': patient['patient_id'],
                'study_id': patient['study_id'],
                'timepoint_weeks': timepoint,
                'measurement_date': measurement_date,
                'efficacy_score': round(score, 1),
                'response_category': 'Responder' if score < baseline_score * 0.5 else 'Non-responder' if timepoint > 0 else 'Baseline'
            })
    
    return pd.DataFrame(records)


def save_datasets(datasets: dict, output_dir: str):
    """Save all datasets to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, df in datasets.items():
        # Save as Parquet
        parquet_file = output_path / f"{name}.parquet"
        df.to_parquet(parquet_file, index=False)
        
        # Also save as CSV for easy viewing
        csv_file = output_path / f"{name}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"Generated {name}: {len(df)} records")
        print(f"  Saved to: {parquet_file}")
        print(f"  CSV copy: {csv_file}")
    
    # Create metadata file
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "datasets": {
            name: {
                "records": len(df),
                "columns": list(df.columns),
                "file_types": ["parquet", "csv"]
            }
            for name, df in datasets.items()
        }
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\nMetadata saved to: {metadata_file}")


def main():
    """Main function to generate all sample data"""
    print("Generating sample clinical trial data...")
    
    # Generate demographics first (other datasets depend on it)
    print("\n1. Generating demographics...")
    demographics = generate_demographics(n_patients=100)
    
    # Generate other datasets
    print("2. Generating vital signs...")
    vital_signs = generate_vital_signs(demographics, n_measurements=5)
    
    print("3. Generating adverse events...")
    adverse_events = generate_adverse_events(demographics)
    
    print("4. Generating laboratory data...")
    laboratory_data = generate_laboratory_data(demographics)
    
    print("5. Generating efficacy data...")
    efficacy_data = generate_efficacy_data(demographics)
    
    # Combine all datasets
    datasets = {
        'demographics': demographics,
        'vital_signs': vital_signs,
        'adverse_events': adverse_events,
        'laboratory_data': laboratory_data,
        'efficacy_data': efficacy_data
    }
    
    # Save datasets
    output_dir = "../data/sample_data"
    save_datasets(datasets, output_dir)
    
    print(f"\n✅ Sample data generation complete!")
    print(f"📁 Output directory: {Path(output_dir).absolute()}")
    print(f"📊 Total datasets: {len(datasets)}")
    print(f"👥 Total patients: {len(demographics)}")
    
    # Print summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"   Vital signs measurements: {len(vital_signs)}")
    print(f"   Adverse events: {len(adverse_events)}")
    print(f"   Laboratory results: {len(laboratory_data)}")
    print(f"   Efficacy measurements: {len(efficacy_data)}")


if __name__ == "__main__":
    main()
