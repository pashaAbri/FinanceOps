"""
Main entry point for HPI forecasting models.
This is the only file that interacts with the outside structure.
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

# Add workflow directory to path
sys.path.append('workflows')
from workflow_1 import HPIForecastingWorkflow


def save_results_to_output(results: Dict[str, Any], filename: str = None):
    """Save results to the output directory."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hpi_forecast_results_{timestamp}.json"
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    
    # Convert results to JSON-serializable format
    json_results = {}
    for key, value in results.items():
        if key == 'raw_data':
            # Skip raw data for JSON output (too large)
            continue
        elif key == 'processed_data':
            # Save summary of processed data
            json_results[key] = {
                'shape': value.shape,
                'columns': list(value.columns),
                'date_range': {
                    'start': value.index.min().isoformat(),
                    'end': value.index.max().isoformat()
                }
            }
        elif key == 'models':
            # Skip model objects (not JSON serializable)
            continue
        else:
            json_results[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"Results saved to: {output_path}")
    return output_path


def run_forecasting_workflow(years: Optional[int] = None, 
                           current_ratio: Optional[float] = None,
                           save_output: bool = True) -> Dict[str, Any]:
    """
    Run the HPI forecasting workflow.
    
    :param years: Number of years for forecasting (default from config)
    :param current_ratio: Current valuation ratio (default: latest from data)
    :param save_output: Whether to save results to output directory
    :return: Dictionary containing all workflow results
    """
    print("Initializing HPI Forecasting Workflow...")
    
    try:
        # Initialize and run workflow
        workflow = HPIForecastingWorkflow()
        results = workflow.run_complete_workflow(years=years, current_ratio=current_ratio)
        
        # Print summary report
        workflow.print_summary_report()
        
        # Save results if requested
        if save_output:
            save_results_to_output(results)
        
        return results
        
    except Exception as e:
        print(f"Error running workflow: {str(e)}")
        raise


def run_quick_forecast(current_ratio: float, years: int = 5) -> Dict[str, float]:
    """
    Run a quick forecast for a specific ratio and time horizon.
    
    :param current_ratio: Current HPI/Earnings ratio
    :param years: Forecast time horizon in years
    :return: Dictionary with forecast results
    """
    print(f"Running quick forecast for ratio {current_ratio:.4f} over {years} years...")
    
    try:
        workflow = HPIForecastingWorkflow()
        
        # Load and preprocess data
        workflow.step_1_load_data()
        workflow.step_2_preprocess_data(years=years)
        
        # Train a single model (basic configuration)
        from modeling.forecast_model import ForecastModel
        
        model = ForecastModel(
            df=workflow.processed_data,
            years=years,
            config_path=workflow.config_path,
            use_mortgage_factor=False,
            use_real_returns=False
        )
        
        # Generate forecast
        mean_return, std_return = model.forecast(current_ratio)
        
        result = {
            'current_ratio': current_ratio,
            'years': years,
            'mean_return': mean_return,
            'std_return': std_return,
            'mean_return_pct': f"{mean_return:.1%}",
            'std_return_pct': f"{std_return:.1%}"
        }
        
        print(f"Forecast: {mean_return:.1%} ± {std_return:.1%} annual return")
        
        return result
        
    except Exception as e:
        print(f"Error running quick forecast: {str(e)}")
        raise


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='HPI Forecasting Model')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Full workflow command
    workflow_parser = subparsers.add_parser('workflow', help='Run complete workflow')
    workflow_parser.add_argument('--years', type=int, help='Forecast years')
    workflow_parser.add_argument('--ratio', type=float, help='Current valuation ratio')
    workflow_parser.add_argument('--no-save', action='store_true', help='Don\'t save results')
    
    # Quick forecast command
    quick_parser = subparsers.add_parser('quick', help='Run quick forecast')
    quick_parser.add_argument('ratio', type=float, help='Current valuation ratio')
    quick_parser.add_argument('--years', type=int, default=5, help='Forecast years (default: 5)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'workflow':
        # Run full workflow
        results = run_forecasting_workflow(
            years=args.years,
            current_ratio=args.ratio,
            save_output=not args.no_save
        )
        
    elif args.command == 'quick':
        # Run quick forecast
        result = run_quick_forecast(args.ratio, args.years)
        
    else:
        # Default: run full workflow
        print("No command specified. Running default workflow...")
        results = run_forecasting_workflow()


if __name__ == "__main__":
    main()
