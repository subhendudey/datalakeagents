#!/usr/bin/env python3.12
"""
Script to run the Clinical Analyst Agent on a dataset
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.agents.clinical_analyst import ClinicalDataAnalyst
    from src.data_lake import DataLakeStorage
    from src.meta_layer import MetadataManager, SemanticModel
    from src.utils.config import settings
except ImportError:
    # Fallback for direct execution
    from agents.clinical_analyst import ClinicalDataAnalyst
    from data_lake import DataLakeStorage
    from meta_layer import MetadataManager, SemanticModel
    from utils.config import settings


def run_clinical_analyst(dataset_id=None, layer="processed"):
    """Run the Clinical Analyst Agent on a dataset"""
    
    print("🔬 Initializing Clinical Analyst Agent...")
    
    # Initialize components
    storage = DataLakeStorage(settings.data_lake.base_path)
    metadata_manager = MetadataManager()
    semantic_model = SemanticModel()
    
    # Initialize the Clinical Analyst Agent
    clinical_analyst = ClinicalDataAnalyst()
    
    # If no dataset_id provided, get first available dataset
    if not dataset_id:
        datasets = storage.list_datasets(layer)
        if layer in datasets and datasets[layer]:
            dataset_id = Path(datasets[layer][0]).stem
            print(f"📋 Using dataset: {dataset_id}")
        else:
            print(f"❌ No datasets found in {layer} layer")
            return
    
    print(f"🎯 Analyzing dataset: {dataset_id}")
    
    try:
        # Load the dataset
        if layer == "curated":
            data = storage.read_curated(f"{dataset_id}.parquet")
        elif layer == "processed":
            # Handle the naming convention
            try:
                data = storage.read_processed(f"{dataset_id}_processed.parquet")
            except:
                data = storage.read_processed(f"{dataset_id}.parquet")
        else:  # raw
            data = storage.read_raw(f"{dataset_id}.parquet")
        
        print(f"📊 Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        print(f"📋 Columns: {list(data.columns)}")
        
        # Get metadata
        metadata = metadata_manager.get_dataset_metadata(dataset_id)
        metadata_dict = metadata.__dict__ if metadata else {}
        
        # Create agent context
        from agents.base_agent import AgentContext
        
        context = AgentContext(
            dataset_id=dataset_id,
            data=data,
            metadata=metadata_dict,
            semantic_model=semantic_model,
            user_query="Perform comprehensive clinical data analysis focusing on patterns, anomalies, and clinical insights"
        )
        
        print("🚀 Running Clinical Analyst Agent...")
        
        # Execute the analysis
        result = clinical_analyst.execute(context)
        
        # Display results
        print("\n" + "="*60)
        print("🎉 CLINICAL ANALYSIS RESULTS")
        print("="*60)
        
        print(f"✅ Analysis Success: {result.success}")
        print(f"📊 Confidence Score: {result.confidence_score:.2f}")
        print(f"⏱️  Execution Time: {result.execution_time:.2f} seconds")
        
        if hasattr(result, 'insights') and result.insights:
            print(f"\n🔍 Key Insights ({len(result.insights)} found):")
            for i, insight in enumerate(result.insights, 1):
                print(f"  {i}. {insight}")
        
        if hasattr(result, 'recommendations') and result.recommendations:
            print(f"\n💡 Recommendations ({len(result.recommendations)}):")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"  {i}. {rec}")
        
        if hasattr(result, 'statistics') and result.statistics:
            print(f"\n📈 Statistical Summary:")
            for key, value in result.statistics.items():
                print(f"  • {key}: {value}")
        
        print("\n" + "="*60)
        print("✅ Clinical Analysis Complete!")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Clinical Analyst Agent")
    parser.add_argument("--dataset-id", help="Dataset ID to analyze")
    parser.add_argument("--layer", default="processed", choices=["raw", "processed", "curated"], 
                       help="Data layer to analyze")
    
    args = parser.parse_args()
    
    # Run the analysis
    result = run_clinical_analyst(args.dataset_id, args.layer)
    
    if result and result.success:
        print("\n🎯 Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
