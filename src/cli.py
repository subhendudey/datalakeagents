#!/usr/bin/env python3
"""
Command Line Interface for Clinical Trials Data Lake

Provides CLI commands for data ingestion, analysis, and management.
"""

import typer
import pandas as pd
from pathlib import Path
from typing import Optional, List
import json
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

try:
    from .data_lake import DataLakeStorage
    from .meta_layer import MetadataManager, SemanticModel
    from .ingestion import IngestionPipeline, IngestionConfig
    from .agents import AgentOrchestrator
    from .utils import settings
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from data_lake import DataLakeStorage
    from meta_layer import MetadataManager, SemanticModel
    from ingestion import IngestionPipeline, IngestionConfig
    from agents import AgentOrchestrator
    from utils.config import settings

app = typer.Typer(help="Clinical Trials Data Lake CLI")
console = Console()


@app.command()
def init(
    base_path: str = typer.Option("./data", help="Base path for data lake"),
    config_file: Optional[str] = typer.Option(None, help="Configuration file path")
):
    """Initialize the data lake"""
    console.print("[bold blue]Initializing Clinical Trials Data Lake...[/bold blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Create storage
        task1 = progress.add_task("Setting up storage layers...", total=None)
        storage = DataLakeStorage(base_path)
        progress.update(task1, description="✅ Storage layers created")
        
        # Initialize metadata manager
        task2 = progress.add_task("Initializing metadata manager...", total=None)
        metadata_manager = MetadataManager()
        progress.update(task2, description="✅ Metadata manager initialized")
        
        # Initialize semantic model
        task3 = progress.add_task("Loading semantic model...", total=None)
        semantic_model = SemanticModel()
        progress.update(task3, description="✅ Semantic model loaded")
        
        # Save semantic model
        task4 = progress.add_task("Saving semantic model...", total=None)
        schemas_dir = f"{base_path}/schemas"
        import os
        os.makedirs(schemas_dir, exist_ok=True)
        semantic_model.export_semantic_model(f"{schemas_dir}/semantic_model.json")
        progress.update(task4, description="✅ Semantic model saved")
    
    console.print(f"[bold green]✅ Data lake initialized successfully at {base_path}[/bold green]")


@app.command()
def ingest(
    file_path: str = typer.Argument(..., help="Path to data file"),
    dataset_name: str = typer.Option(..., help="Dataset name"),
    dataset_description: str = typer.Option(..., help="Dataset description"),
    domain: str = typer.Option(..., help="Clinical domain"),
    created_by: str = typer.Option("cli_user", help="User who created the dataset"),
    config_file: Optional[str] = typer.Option(None, help="Configuration file path")
):
    """Ingest a data file into the data lake"""
    console.print(f"[bold blue]Ingesting {file_path}...[/bold blue]")
    
    # Initialize components
    storage = DataLakeStorage(settings.data_lake.base_path)
    metadata_manager = MetadataManager()
    semantic_model = SemanticModel()
    
    # Create ingestion pipeline
    config = IngestionConfig()
    pipeline = IngestionPipeline(storage, metadata_manager, semantic_model, config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task = progress.add_task("Processing ingestion pipeline...", total=None)
        
        # Ingest file
        result = pipeline.ingest_file(
            file_path=file_path,
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            domain=domain,
            created_by=created_by
        )
        
        progress.update(task, description="✅ Ingestion completed")
    
    # Display results
    console.print(f"\n[bold]Ingestion Results:[/bold]")
    console.print(f"Status: {result.status.value}")
    console.print(f"Dataset ID: {result.dataset_id}")
    console.print(f"Records Processed: {result.records_processed}")
    console.print(f"Records Rejected: {result.records_rejected}")
    console.print(f"Quality Score: {result.quality_score:.2f}")
    console.print(f"Processing Time: {result.processing_time:.2f}s")
    
    if result.validation_errors:
        console.print(f"\n[red]Validation Errors:[/red]")
        for error in result.validation_errors:
            console.print(f"  • {error}")
    
    if result.transformation_warnings:
        console.print(f"\n[yellow]Transformation Warnings:[/yellow]")
        for warning in result.transformation_warnings:
            console.print(f"  • {warning}")
    
    if result.created_datasets:
        console.print(f"\n[green]Created Datasets:[/green]")
        for dataset in result.created_datasets:
            console.print(f"  • {dataset}")


@app.command()
def analyze(
    dataset_id: str = typer.Argument(..., help="Dataset ID to analyze"),
    query: Optional[str] = typer.Option(None, help="Analysis query"),
    agents: Optional[List[str]] = typer.Option(None, help="Specific agents to use"),
    config_file: Optional[str] = typer.Option(None, help="Configuration file path")
):
    """Analyze a dataset using AI agents"""
    console.print(f"[bold blue]Analyzing dataset {dataset_id}...[/bold blue]")
    
    # Initialize components
    storage = DataLakeStorage(settings.data_lake.base_path)
    metadata_manager = MetadataManager()
    semantic_model = SemanticModel()
    orchestrator = AgentOrchestrator()
    
    # Load dataset
    try:
        # Try to load from curated layer first
        try:
            data = storage.read_curated(f"{dataset_id}.parquet")
        except:
            # Fallback to processed layer with correct naming convention
            try:
                data = storage.read_processed(f"{dataset_id}_processed.parquet")
            except:
                # Try without suffix as fallback
                data = storage.read_processed(f"{dataset_id}.parquet")
        
        metadata = metadata_manager.get_dataset_metadata(dataset_id).__dict__ if metadata_manager.get_dataset_metadata(dataset_id) else {}
        
    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task = progress.add_task("Running AI analysis...", total=None)
        
        # Run analysis
        results = orchestrator.analyze_dataset(
            dataset_id=dataset_id,
            data=data,
            metadata=metadata,
            semantic_model=semantic_model,
            user_query=query,
            preferred_agents=agents
        )
        
        progress.update(task, description="✅ Analysis completed")
    
    # Display results
    console.print(f"\n[bold]Analysis Results:[/bold]")
    console.print(f"Success: {results['success']}")
    console.print(f"Total Agents: {results['summary']['total_agents']}")
    console.print(f"Successful Agents: {results['summary']['successful_agents']}")
    console.print(f"Overall Confidence: {results['overall_confidence']:.2f}")
    console.print(f"Execution Time: {results['execution_time']:.2f}s")
    
    if results['insights']:
        console.print(f"\n[bold green]Key Insights:[/bold green]")
        for insight in results['insights'][:10]:  # Show top 10
            console.print(f"  • {insight}")
    
    if results['recommendations']:
        console.print(f"\n[bold yellow]Recommendations:[/bold yellow]")
        for rec in results['recommendations'][:10]:  # Show top 10
            console.print(f"  • {rec}")
    
    # Show agent-specific results
    console.print(f"\n[bold]Agent Results:[/bold]")
    table = Table(title="Agent Performance")
    table.add_column("Agent", style="cyan")
    table.add_column("Success", style="green")
    table.add_column("Confidence", style="blue")
    table.add_column("Insights", style="magenta")
    
    for agent_name, agent_result in results['agent_results'].items():
        table.add_row(
            agent_name,
            "✅" if agent_result['success'] else "❌",
            f"{agent_result['confidence_score']:.2f}",
            str(len(agent_result['insights']))
        )
    
    console.print(table)


@app.command()
def analyze_all(
    query: Optional[str] = typer.Option(None, help="Analysis query to apply to all datasets"),
    agents: Optional[List[str]] = typer.Option(None, help="Specific agents to use"),
    layer: Optional[str] = typer.Option("processed", help="Layer to analyze (raw, processed, curated)"),
    config_file: Optional[str] = typer.Option(None, help="Configuration file path"),
    basic: bool = typer.Option(True, help="Use basic analysis instead of AI agents")
):
    """Analyze all datasets in the data lake"""
    console.print("[bold blue]Analyzing all datasets...[/bold blue]")
    
    # Initialize components
    storage = DataLakeStorage(settings.data_lake.base_path)
    metadata_manager = MetadataManager()
    
    # Get all datasets from specified layer
    datasets = storage.list_datasets(layer)
    dataset_files = []
    
    # Collect dataset files from the specified layer
    if layer in datasets:
        dataset_files = datasets[layer]
    else:
        console.print(f"[red]No datasets found in layer '{layer}'[/red]")
        raise typer.Exit(1)
    
    if not dataset_files:
        console.print(f"[yellow]No datasets found in {layer} layer[/yellow]")
        return
    
    console.print(f"[bold]Found {len(dataset_files)} datasets in {layer} layer[/bold]")
    
    # Analyze each dataset
    successful_analyses = 0
    failed_analyses = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        for i, dataset_path in enumerate(dataset_files):
            dataset_name = Path(dataset_path).stem
            task = progress.add_task(f"Analyzing {dataset_name}... ({i+1}/{len(dataset_files)})", total=None)
            
            try:
                # Load dataset
                try:
                    if layer == "curated":
                        data = storage.read_curated(dataset_path)
                    elif layer == "processed":
                        # Fix path issue - remove duplicate layer prefix
                        clean_path = dataset_path.replace(f"{layer}/", "") if dataset_path.startswith(f"{layer}/") else dataset_path
                        data = storage.read_processed(clean_path)
                    else:  # raw
                        data = storage.read_raw(dataset_path)
                except Exception as e:
                    console.print(f"[yellow]⚠️  Could not load {dataset_name}: {e}[/yellow]")
                    failed_analyses += 1
                    continue
                
                # Get metadata
                metadata = metadata_manager.get_dataset_metadata(dataset_name)
                
                # Run analysis
                try:
                    if basic:
                        # Basic statistical analysis
                        results = _basic_dataset_analysis(dataset_name, data, metadata)
                        console.print(f"[green]✅ {dataset_name}: Basic analysis completed[/green]")
                        successful_analyses += 1
                        
                        # Display brief results
                        console.print(f"   📊 {results['summary']}")
                    else:
                        # Try AI agent analysis
                        semantic_model = SemanticModel()
                        orchestrator = AgentOrchestrator()
                        
                        metadata_dict = metadata.__dict__ if metadata else {}
                        
                        results = orchestrator.analyze_dataset(
                            dataset_id=dataset_name,
                            data=data,
                            metadata=metadata_dict,
                            semantic_model=semantic_model,
                            user_query=query,
                            preferred_agents=agents
                        )
                        
                        console.print(f"[green]✅ {dataset_name}: AI analysis completed[/green]")
                        successful_analyses += 1
                        
                        # Display brief results
                        if 'summary' in results:
                            console.print(f"   📊 {results['summary']}")
                    
                except Exception as e:
                    console.print(f"[red]❌ {dataset_name}: Analysis failed - {e}[/red]")
                    failed_analyses += 1
                
            except Exception as e:
                console.print(f"[red]❌ {dataset_name}: Error - {e}[/red]")
                failed_analyses += 1
            
            progress.update(task, description=f"Completed {dataset_name}")
    
    # Summary
    console.print(f"\n[bold]🎯 Analysis Summary:[/bold]")
    console.print(f"✅ Successful analyses: {successful_analyses}")
    console.print(f"❌ Failed analyses: {failed_analyses}")
    console.print(f"📊 Success rate: {successful_analyses/(successful_analyses+failed_analyses)*100:.1f}%" if (successful_analyses+failed_analyses) > 0 else "N/A")


def _basic_dataset_analysis(dataset_name: str, data: pd.DataFrame, metadata) -> dict:
    """Perform basic statistical analysis on a dataset"""
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    analysis = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "shape": data.shape,
        "columns": list(data.columns),
        "data_types": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "summary_stats": {}
    }
    
    # Basic statistics for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        analysis["summary_stats"] = data[numeric_cols].describe().to_dict()
    
    # Basic statistics for categorical columns
    categorical_cols = data.select_dtypes(include=['object', 'string']).columns
    if len(categorical_cols) > 0:
        analysis["categorical_stats"] = {}
        for col in categorical_cols:
            analysis["categorical_stats"][col] = {
                "unique_count": data[col].nunique(),
                "top_values": data[col].value_counts().head().to_dict()
            }
    
    # Create summary
    summary_parts = [
        f"{data.shape[0]} rows, {data.shape[1]} columns",
        f"{len(numeric_cols)} numeric, {len(categorical_cols)} categorical"
    ]
    
    if metadata:
        summary_parts.append(f"Domain: {getattr(metadata, 'domain', 'Unknown')}")
        if hasattr(metadata, 'row_count'):
            summary_parts.append(f"Records: {metadata.row_count}")
    
    analysis["summary"] = " | ".join(summary_parts)
    
    return analysis


@app.command()
def list_datasets(
    layer: Optional[str] = typer.Option(None, help="Filter by layer (raw, processed, curated)"),
    domain: Optional[str] = typer.Option(None, help="Filter by domain")
):
    """List datasets in the data lake"""
    console.print("[bold blue]Listing datasets...[/bold blue]")
    
    # Initialize components
    storage = DataLakeStorage(settings.data_lake.base_path)
    metadata_manager = MetadataManager()
    
    # Get datasets from storage
    datasets = storage.list_datasets(layer)
    
    # Create table for display
    table = Table(title="Datasets")
    table.add_column("Layer", style="cyan")
    table.add_column("Dataset Name", style="magenta")
    table.add_column("File Type", style="green")
    table.add_column("Size", style="yellow")
    table.add_column("Metadata", style="blue")
    
    # Get metadata and display datasets
    all_metadata = []
    for layer_name, dataset_list in datasets.items():
        for dataset_path in dataset_list:
            dataset_name = Path(dataset_path).stem
            file_extension = Path(dataset_path).suffix
            
            # Try to get metadata
            metadata = metadata_manager.get_dataset_metadata(dataset_name)
            metadata_status = "✅ Available" if metadata else "❌ Not available"
            
            # Get file size
            try:
                file_size = Path(f"./data/{layer_name}/{dataset_path}").stat().st_size
                size_str = f"{file_size:,} bytes"
            except:
                size_str = "Unknown"
            
            # Add to table
            table.add_row(
                layer_name.capitalize(),
                dataset_name,
                file_extension,
                size_str,
                metadata_status
            )
            
            if metadata:
                all_metadata.append((layer_name, metadata))
    
    # Filter by domain if specified (only works for datasets with metadata)
    if domain and all_metadata:
        filtered_metadata = [(l, m) for l, m in all_metadata if m.domain == domain]
        if filtered_metadata:
            console.print(f"\n[bold]Datasets in domain '{domain}':[/bold]")
            for layer_name, metadata in filtered_metadata:
                console.print(f"  • {metadata.name} ({layer_name}) - {metadata.description}")
        else:
            console.print(f"[yellow]No datasets found in domain '{domain}'[/yellow]")
    
    # Display the table
    console.print(table)
    
    # Summary
    total_files = sum(len(dataset_list) for dataset_list in datasets.values())
    console.print(f"\n[bold]Total files: {total_files}[/bold]")
    console.print(f"[bold]With metadata: {len(all_metadata)}[/bold]")


@app.command()
def generate_sample_data(
    output_dir: str = typer.Option("./data/sample_data", help="Output directory"),
    n_patients: int = typer.Option(100, help="Number of patients to generate")
):
    """Generate sample clinical trial data"""
    console.print(f"[bold blue]Generating sample data for {n_patients} patients...[/bold blue]")
    
    # Import the sample data generator
    import sys
    import os
    script_path = Path(__file__).parent.parent / "scripts" / "generate_sample_data.py"
    
    # Run the script
    import subprocess
    result = subprocess.run([sys.executable, str(script_path)], 
                          cwd=Path(__file__).parent.parent,
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        console.print("[bold green]✅ Sample data generated successfully[/bold green]")
        console.print(result.stdout)
    else:
        console.print(f"[red]Error generating sample data: {result.stderr}[/red]")
        raise typer.Exit(1)


@app.command()
def ingest_all():
    """Ingest all sample data files into the data lake"""
    console.print("[bold blue]Starting batch ingestion of all sample data files...[/bold blue]")
    
    import subprocess
    import sys
    
    try:
        # Run the batch ingestion script
        result = subprocess.run([
            sys.executable, "scripts/ingest_all_data.py"
        ], capture_output=False, text=True, cwd=".")
        
        if result.returncode == 0:
            console.print("[bold green]✅ Batch ingestion completed successfully![/bold green]")
        else:
            console.print("[bold red]❌ Batch ingestion failed[/bold red]")
            
    except Exception as e:
        console.print(f"[bold red]❌ Error running batch ingestion: {e}[/bold red]")


@app.command()
def status():
    """Show data lake status"""
    console.print("[bold blue]Data Lake Status[/bold blue]")
    
    # Initialize components
    storage = DataLakeStorage(settings.data_lake.base_path)
    metadata_manager = MetadataManager()
    
    # Get storage info
    datasets = storage.list_datasets()
    total_datasets = sum(len(ds_list) for ds_list in datasets.values())
    
    # Get metadata info
    all_metadata = metadata_manager.datasets
    
    # Display status
    console.print(f"\n[bold]Storage Information:[/bold]")
    console.print(f"Base Path: {settings.data_lake.base_path}")
    console.print(f"Total Datasets: {total_datasets}")
    
    for layer_name, dataset_list in datasets.items():
        console.print(f"  {layer_name.capitalize()}: {len(dataset_list)} datasets")
    
    console.print(f"\n[bold]Metadata Information:[/bold]")
    console.print(f"Total Metadata Records: {len(all_metadata)}")
    
    # Quality distribution
    quality_counts = {}
    for metadata in all_metadata.values():
        quality = metadata.data_quality_level.value
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    console.print(f"\n[bold]Data Quality Distribution:[/bold]")
    for quality, count in quality_counts.items():
        console.print(f"  {quality.capitalize()}: {count} datasets")


def main():
    """Main CLI entry point"""
    app()


if __name__ == "__main__":
    main()
