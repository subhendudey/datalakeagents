#!/usr/bin/env python3.12
"""
Script to ingest all sample data files into the data lake
"""

import subprocess
import sys
from pathlib import Path

# Define the sample data files with their metadata
# Get the project root directory (where this script is located)
PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample_data"

SAMPLE_FILES = [
    {
        "file": str(SAMPLE_DATA_DIR / "demographics.csv"),
        "name": "Sample Demographics",
        "description": "Generated sample demographics data",
        "domain": "demographics"
    },
    {
        "file": str(SAMPLE_DATA_DIR / "adverse_events.csv"),
        "name": "Sample Adverse Events",
        "description": "Generated sample adverse events data",
        "domain": "safety"
    },
    {
        "file": str(SAMPLE_DATA_DIR / "efficacy_data.csv"),
        "name": "Sample Efficacy Data",
        "description": "Generated sample efficacy data",
        "domain": "efficacy"
    },
    {
        "file": str(SAMPLE_DATA_DIR / "laboratory_data.csv"),
        "name": "Sample Laboratory Data",
        "description": "Generated sample laboratory data",
        "domain": "laboratory"
    },
    {
        "file": str(SAMPLE_DATA_DIR / "vital_signs.csv"),
        "name": "Sample Vital Signs",
        "description": "Generated sample vital signs data",
        "domain": "vital_signs"
    }
]

def ingest_file(file_info):
    """Ingest a single file"""
    cmd = [
        sys.executable, "src/cli.py", "ingest",
        file_info["file"],
        "--dataset-name", file_info["name"],
        "--dataset-description", file_info["description"],
        "--domain", file_info["domain"],
        "--created-by", "batch_ingest"
    ]
    
    print(f"\n{'='*60}")
    print(f"Ingesting: {file_info['name']}")
    print(f"File: {file_info['file']}")
    print(f"Domain: {file_info['domain']}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ SUCCESS: File ingested successfully")
            # Extract key information from output
            lines = result.stdout.split('\n')
            for line in lines:
                if "Dataset ID:" in line:
                    print(f"📋 {line.strip()}")
                elif "Records Processed:" in line:
                    print(f"📊 {line.strip()}")
                elif "Quality Score:" in line:
                    print(f"📈 {line.strip()}")
                elif "Status: completed" in line:
                    print(f"🎉 {line.strip()}")
        else:
            print(f"❌ FAILED: Return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("-" * 60)

def main():
    """Main function to ingest all files"""
    print("🚀 Starting batch ingestion of all sample data files...")
    
    # Check if we're in the right directory
    if not Path("src/cli.py").exists():
        print("❌ ERROR: Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if sample data exists
    for file_info in SAMPLE_FILES:
        if not Path(file_info["file"]).exists():
            print(f"❌ ERROR: Sample file not found: {file_info['file']}")
            sys.exit(1)
    
    # Ingest each file
    success_count = 0
    total_count = len(SAMPLE_FILES)
    
    for file_info in SAMPLE_FILES:
        ingest_file(file_info)
        success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🎯 BATCH INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total files processed: {success_count}/{total_count}")
    print(f"Success rate: {success_count/total_count*100:.1f}%")
    
    # Show final status
    print(f"\n📊 Final Data Lake Status:")
    try:
        result = subprocess.run([
            sys.executable, "src/cli.py", "status"
        ], capture_output=True, text=True, cwd=".")
        print(result.stdout)
    except Exception as e:
        print(f"Could not get final status: {e}")

if __name__ == "__main__":
    main()
