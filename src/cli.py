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

from .data_lake import DataLakeStorage
from .meta_layer import MetadataManager, SemanticModel
from .ingestion import IngestionPipeline, IngestionConfig
from .agents import AgentOrchestrator
from .utils import settings

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
        semantic_model.export_semantic_model(f"{base_path}/schemas/semantic_model.json")
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
            # Fallback to processed layer
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
    
    # Get metadata
    all_metadata = []
    for layer_name, dataset_list in datasets.items():
        for dataset_path in dataset_list:
            dataset_name = Path(dataset_path).stem
            metadata = metadata_manager.get_dataset_metadata(dataset_name)
            if metadata:
                all_metadata.append((layer_name, metadata))
    
    # Filter by domain if specified
    if domain:
        all_metadata = [(l, m) for l, m in all_metadata if m.domain == domain]
    
    # Display results
    if not all_metadata:
        console.print("[yellow]No datasets found[/yellow]")
        return
    
    table = Table(title="Datasets")
    table.add_column("Layer", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Domain", style="blue")
    table.add_column("Records", style="magenta")
    table.add_column("Quality", style="yellow")
    table.add_column("Created", style="red")
    
    for layer_name, metadata in all_metadata:
        table.add_row(
            layer_name,
            metadata.name,
            metadata.domain,
            str(metadata.row_count) if metadata.row_count else "N/A",
            metadata.data_quality_level.value,
            metadata.created_at.strftime("%Y-%m-%d") if metadata.created_at else "N/A"
        )
    
    console.print(table)


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
