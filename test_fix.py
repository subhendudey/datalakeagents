#!/usr/bin/env python3.12
import pandas as pd

# Test the issue with string columns
data = pd.DataFrame({
    'patient_id': [1, 2, 3],
    'age': [25, 30, 35],
    'gender': ['M', 'F', 'M']
})

# Test the problematic line
try:
    # This should work now
    result = data.mean()
    print(f"Mean calculation successful: {result}")
except Exception as e:
    print(f"Error: {e}")
