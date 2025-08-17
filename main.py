#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Target Lipid-Cholesterol Rich Domain Transport Analysis

Main entry point for target lipid-mediated cholesterol-rich domain transport analysis.
This script coordinates the full analysis pipeline including trajectory processing,
Hidden Markov Model analysis, Bayesian statistics, and figure generation.

Usage:
    python main.py [--output DIR] [--n-jobs N] [--force] [--target-lipid LIPID] [--config CONFIG] [--topology PSF] [--trajectory XTC]

Output:
    Three publication figures:
    - state_distribution.png/svg: Protein state distributions
    - {target_lipid}_effect_summary.png/svg: Target lipid transport and conditional effects  
    - group_effect_size.png/svg: Group-level Bayesian effect analysis
"""

import argparse
import os
import sys
import time
import shutil
import logging

# Import the analysis pipeline and configuration
from src.analysis_pipeline import run_analysis_pipeline
from src.config import config, load_config_from_file


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('GM3Analysis')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Target Lipid-CS Rich Domain Interaction Analysis - Publication Figures Only'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=f'publication_figures_{time.strftime("%Y%m%d_%H%M%S")}',
        help='Output directory for results'
    )
    parser.add_argument(
        '--n-jobs', 
        type=int, 
        default=-1,
        help='Number of parallel jobs (-1 for all CPUs)'
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force overwrite of existing output directory'
    )
    parser.add_argument(
        '--target-lipid',
        type=str,
        default=None,
        help='Target lipid for analysis (e.g., GM3, DPG3, DPGS). If not specified, uses config default.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (JSON format)'
    )
    parser.add_argument(
        '--topology',
        type=str,
        default=None,
        help='Path to topology file (PSF format)'
    )
    parser.add_argument(
        '--trajectory',
        type=str,
        default=None,
        help='Path to trajectory file (XTC/DCD format)'
    )
    return parser.parse_args()


def setup_output_directory(output_dir, force=False):
    """Set up the output directory, handling existing directories."""
    if os.path.exists(output_dir):
        if force:
            print(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"Output directory {output_dir} already exists. Use --force to overwrite.")
            response = input("Overwrite existing directory? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(output_dir)
            else:
                print("Aborting.")
                sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir


def main():
    """Main entry point for the target lipid domain transport analysis."""
    logger = setup_logging()
    args = parse_arguments()
    
    # Load configuration file if provided
    if args.config:
        print(f"Loading configuration from: {args.config}")
        if not load_config_from_file(args.config):
            print("Failed to load configuration file. Using defaults.")
    
    # Set target lipid if provided
    if args.target_lipid:
        config.set_target_lipid(args.target_lipid)
        print(f"Target lipid set to: {config.TARGET_LIPID}")
    
    # Set trajectory files if provided
    if args.topology:
        config.TOPOLOGY_FILE = args.topology
        print(f"Topology file set to: {config.TOPOLOGY_FILE}")
    
    if args.trajectory:
        config.TRAJECTORY_FILE = args.trajectory
        print(f"Trajectory file set to: {config.TRAJECTORY_FILE}")
    
    # Print current configuration
    config.print_config()
    
    # Set up output directory
    output_dir = setup_output_directory(args.output, args.force)
    
    # Run the analysis pipeline
    try:
        logger.info(f"Starting {config.TARGET_LIPID} domain transport analysis...")
        all_results = run_analysis_pipeline(output_dir, args.n_jobs)
        
        if all_results:
            print(f"\nAnalysis completed successfully!")
            print(f"Publication figures saved to: {output_dir}")
            print("\nGenerated files:")
            print(f"- state_distribution.png/svg")
            print(f"- {config.TARGET_LIPID.lower()}_effect_summary.png/svg")
            if 'protein_data' in all_results and len(all_results['protein_data']) > 1:
                print(f"- group_effect_size.png/svg")
        else:
            print("Analysis failed.")
            return 1
            
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())