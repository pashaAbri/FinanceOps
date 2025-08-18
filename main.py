#!/usr/bin/env python3
"""
Main entry point for the HPI Forecasting system.

This module provides the primary interface for running HPI forecasting workflows
and models from the command line or as a Python module.

Usage:
    python main.py                    # Run default workflow
    python main.py workflow --years 10 --ratio 2.5
    python main.py quick 2.3 --years 5
    
    Or import as a module:
    from models.workflows import HPIForecastingWorkflow
    workflow = HPIForecastingWorkflow()
    results = workflow.run_complete_workflow()
"""

import sys
from pathlib import Path

# Ensure the forecasting_hpi package is importable
package_root = Path(__file__).parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

# Import the main run module
from models.run import main as run_main

def main():
    """Main entry point that delegates to the models.run module."""
    try:
        run_main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
