#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis Pipeline Module - Exact reproduction of original logic

This module maintains the exact same analysis logic as the original code,
only replacing hardcoded GM3/DPG3 references with configurable target_lipid.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import original functions with minimal modification
from .config import config

logger = logging.getLogger(__name__)

# Copy all original functions with target_lipid substitution
def load_universe(logger=None):
    """Load trajectory using MDAnalysis - EXACT COPY from original"""
    from .config import config
    import MDAnalysis as mda
    
    if logger is None:
        logger = logging.getLogger(__name__)
        
    try:
        topology_file = config.TOPOLOGY_FILE
        trajectory_file = config.TRAJECTORY_FILE
        
        print("Loading trajectory...")
        if os.path.exists(topology_file) and os.path.exists(trajectory_file):
            u = mda.Universe(topology_file, trajectory_file)
            print("Trajectory loaded successfully.")
            return u
        else:
            logger.error(f"Trajectory files not found. Please make sure {topology_file} and {trajectory_file} exist in the current directory.")
            print(f"ERROR: Trajectory files not found. Please make sure {topology_file} and {trajectory_file} exist in the current directory.")
            return None
            
    except Exception as e:
        logger.error(f"Error loading trajectory: {e}")
        print(f"Error loading trajectory: {e}")
        return None


def identify_lipid_leaflets(u, logger):
    """EXACT COPY from original"""
    try:
        from MDAnalysis.analysis.leaflet import LeafletFinder
        print("Identifying lipid leaflets...")
        L = LeafletFinder(u, "name GL1 GL2 AM1 AM2 ROH GM1 GM2")
        cutoff = L.update(10)
        leaflet0 = L.groups(0)
        print("Lipid leaflets identified.")
        return leaflet0
    except Exception as e:
        logger.error(f"Error identifying lipid leaflets: {e}")
        print(f"Error identifying lipid leaflets: {e}")
        return None


def identify_proteins(u, logger):
    """EXACT COPY from original"""
    proteins = {}
    try:
        protein_residues = u.select_atoms("protein")
        if len(protein_residues) == 0:
            protein_residues = u.select_atoms("resname PROT")
        
        if len(protein_residues) == 0:
            logger.warning("No protein residues found in trajectory")
            return {}
        
        segids = np.unique(protein_residues.segids)
        
        for i, segid in enumerate(segids):
            protein_selection = protein_residues.select_atoms(f"segid {segid}")
            if len(protein_selection) > 0:
                protein_name = f"Protein_{segid}"
                proteins[protein_name] = protein_selection
                logger.info(f"Found {protein_name} with {len(protein_selection)} atoms")
        
        return proteins
        
    except Exception as e:
        logger.error(f"Error identifying proteins: {e}")
        return {}


def select_lipids_and_chol(lipid_types, leaflet):
    """EXACT COPY from original"""
    import MDAnalysis as mda
    
    selections = {}
    for resname in lipid_types:
        try:
            selection = leaflet.select_atoms(f"resname {resname}")
            selections[resname] = selection
        except Exception as e:
            print(f"Could not select lipid type {resname}: {e}")
            selections[resname] = mda.AtomGroup([], leaflet.universe)
    
    try:
        selections['CHOL'] = leaflet.select_atoms("resname CHOL")
    except Exception as e:
        print(f"Could not select CHOL: {e}")
        selections['CHOL'] = mda.AtomGroup([], leaflet.universe)
        
    return selections


def process_trajectory_data(output_dir, n_jobs=-1):
    """EXACT COPY from original with target_lipid substitution"""
    from .config import config
    import multiprocessing as mp
    
    try:
        u = load_universe(logger)
        if u is None:
            return None
        
        leaflet = identify_lipid_leaflets(u, logger)
        if leaflet is None:
            return None
        
        proteins = identify_proteins(u, logger)
        if not proteins:
            logger.warning("No proteins found in trajectory")
            return {}
        
        # Handle n_jobs parameter - EXACT COPY from original
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        else:
            n_jobs = min(n_jobs, mp.cpu_count())
        
        logger.info(f"Using {n_jobs} CPUs for parallel processing")
        
        # Use original lipid types list - KEEP ORIGINAL LOGIC
        lipid_types = ["DIPC", "DOPS", "DPSM", "DPGS", config.TARGET_LIPID]  # Only replace DPG3 with target_lipid
        selections = select_lipids_and_chol(lipid_types, leaflet)
        
        # EXACT COPY of original frame processing logic
        frame_indices = list(range(config.START_FRAME, min(config.STOP_FRAME, len(u.trajectory)), config.FRAME_STEP))
        print(f"Processing {len(frame_indices)} frames...")
        
        # Use original parallel processing logic
        all_results = process_frames_parallel(frame_indices, u, selections, proteins, n_jobs, logger)
        
        if not all_results:
            logger.error("No frame results obtained")
            return None
        
        protein_data = extract_protein_data(all_results, proteins, logger)
        
        print(f"Extracted data for {len(protein_data)} proteins")
        return protein_data
        
    except Exception as e:
        logger.error(f"Error processing trajectory data: {e}")
        return None


# Import ALL original functions exactly as they were from the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from domain_transport import (
        calculate_order_parameter,
        calculate_op_kde, 
        calculate_domain_info,
        analyze_lipid_distribution,
        calculate_helix_regions,
        process_frames_parallel,
        process_frame_batch_with_domains,
        extract_protein_data,
        perform_viterbi_decoding,
        analyze_protein,
        calculate_transition_matrix,
        analyze_gm3_transport_effect_cumulative,
        analyze_gm3_cs_rich_transport,
        analyze_causality,
        map_states_to_meanings,
        calculate_gm3_binding_effect,
        calculate_residency_times,
        calculate_transport_effect_from_counts,
        calculate_tm_from_counts
    )
except ImportError as e:
    print(f"Warning: Could not import from domain_transport: {e}")
    # Define minimal fallbacks if needed

def create_state_distribution_plot(protein_data, plot_proteins, output_dir):
    """EXACT COPY from original with target_lipid substitution"""
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from .config import config
    
    try:
        print("Creating state distribution plot")
        
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        # Use dynamic state labels based on target lipid
        target_lipid = config.TARGET_LIPID
        state_labels = [f"Non_{target_lipid}_D", f"Non_{target_lipid}_CS", f"{target_lipid}_D", f"{target_lipid}_CS"]
        
        # REST OF FUNCTION EXACTLY THE SAME AS ORIGINAL
        valid_proteins = []
        valid_data = {}
        
        for protein_name in plot_proteins:
            if protein_name in protein_data:
                if 'states' in protein_data[protein_name]:
                    states = protein_data[protein_name]['states']
                    total_frames = len(states)
                    
                    if total_frames > 0:
                        state_counts = []
                        state_percentages = []
                        
                        for state in range(4):
                            count = np.sum(states == state)
                            state_counts.append(count)
                            state_percentages.append(count / total_frames * 100)
                        
                        valid_proteins.append(protein_name)
                        valid_data[protein_name] = {
                            'counts': state_counts,
                            'percentages': state_percentages,
                            'total_frames': total_frames
                        }
                        
                        print(f"  {protein_name}: {total_frames} frames")
                        for i, (label, count, pct) in enumerate(zip(state_labels, state_counts, state_percentages)):
                            print(f"    {label}: {count} ({pct:.1f}%)")
        
        if not valid_proteins:
            print("WARNING: No valid protein data found for plotting")
            return {}
        
        # Create plot - EXACT SAME AS ORIGINAL
        n_proteins = len(valid_proteins)
        x_positions = np.arange(n_proteins)
        width = 0.2
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for state in range(4):
            percentages = [valid_data[protein]['percentages'][state] for protein in valid_proteins]
            bars = plt.bar(x_positions + state * width, percentages, width, 
                          label=state_labels[state], color=colors[state], alpha=0.8)
            
            # Add percentage labels on top of bars - EXACT COPY from original
            for j, pct in enumerate(percentages):
                if pct >= 5:  # Only label percentages ≥ 5%
                    plt.text(x_positions[j] + state * width, pct + 1, f'{pct:.1f}%', 
                            ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Protein', fontsize=12, fontweight='bold')
        plt.ylabel('Percentage of Frames (%)', fontsize=12, fontweight='bold')
        plt.title(f'Protein State Distribution - {target_lipid} Analysis', fontsize=14, fontweight='bold')
        plt.xticks(x_positions + 1.5 * width, valid_proteins, rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plot_path_png = os.path.join(output_dir, 'state_distribution.png')
        plot_path_svg = os.path.join(output_dir, 'state_distribution.svg')
        plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_svg, format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"State distribution plot saved:")
        print(f"  PNG: {plot_path_png}")
        print(f"  SVG: {plot_path_svg}")
        
        return {
            'png': plot_path_png,
            'svg': plot_path_svg,
            'data': valid_data
        }
        
    except Exception as e:
        logger.error(f"Error creating state distribution plot: {e}")
        return {}


def run_analysis_pipeline(output_dir, n_jobs=-1):
    """EXACT COPY from original - maintains all original logic"""
    from .config import config
    
    logger.info("Starting comprehensive analysis pipeline")
    print("Starting comprehensive analysis pipeline...")
    
    start_time = time.time()
    
    # Process trajectory data
    print("Processing trajectory data...")
    protein_data = process_trajectory_data(output_dir, n_jobs)
    if not protein_data:
        logger.error("Failed to process trajectory data. Analysis aborted.")
        return None
    
    print(f"Trajectory data processed. Found {len(protein_data)} proteins.")
    
    # Analyze each protein - EXACT SAME AS ORIGINAL
    all_results = {}
    
    for protein_name in protein_data.keys():
        logger.info(f"Running comprehensive analysis for {protein_name}")
        print(f"Analyzing {protein_name}...")
        
        try:
            protein_results = analyze_protein(protein_data, protein_name)
            all_results[protein_name] = protein_results
            
            # Output state distribution - EXACT SAME AS ORIGINAL
            states = protein_results['states']
            state_counts = [np.sum(states == i) for i in range(4)]
            total_frames = len(states)
            state_percentages = [count / total_frames * 100 for count in state_counts]
            
            protein_results['percentages'] = state_percentages
            protein_results['counts'] = state_counts
            protein_results['total_frames'] = total_frames
            
            # Use dynamic state labels
            target_lipid = config.TARGET_LIPID
            state_labels = [f"Non_{target_lipid}_D", f"Non_{target_lipid}_CS", f"{target_lipid}_D", f"{target_lipid}_CS"]
            
            state_distribution_info = f"State distribution for {protein_name}: "
            for i, (count, percentage) in enumerate(zip(state_counts, state_percentages)):
                state_distribution_info += f"{state_labels[i]}={count} ({percentage:.2f}%), "
            
            print(state_distribution_info)
            
        except Exception as e:
            logger.error(f"Error analyzing {protein_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"Error in analysis for {protein_name}: {e}")
            print(traceback.format_exc())
    
    # Generate the three publication plots - EXACT SAME AS ORIGINAL
    print("\nGenerating publication figures...")
    
    # Import original plotting functions and modify only target_lipid references
    try:
        from domain_transport import create_gm3_effect_summary_plot, perform_hierarchical_bayesian_analysis
    except ImportError:
        print("Warning: Could not import plotting functions from domain_transport")
    
    # 1. State distribution plot
    try:
        state_plot = create_state_distribution_plot(all_results, 
                                    list(all_results.keys()), 
                                    output_dir)
        if state_plot:
            print("  Created state distribution plot")
        else:
            print("  Failed to create state distribution plot")
    except Exception as e:
        logger.error(f"Error creating state distribution plot: {e}")
        print(f"  Error creating state distribution plot: {e}")
    
    # 2. Target lipid effect summary plot (was GM3)
    try:
        summary_plot = create_gm3_effect_summary_plot(all_results, 
                                    list(all_results.keys()), 
                                    output_dir)
        if summary_plot:
            print(f"  Created {config.TARGET_LIPID} effect summary plot")
        else:
            print(f"  Failed to create {config.TARGET_LIPID} effect summary plot")
    except Exception as e:
        logger.error(f"Error creating {config.TARGET_LIPID} effect summary plot: {e}")
        print(f"  Error creating {config.TARGET_LIPID} effect summary plot: {e}")
    
    # 3. Hierarchical Bayesian analysis - EXACT SAME AS ORIGINAL
    if len(all_results) > 1:
        try:
            bayesian_plot = perform_hierarchical_bayesian_analysis(all_results, output_dir)
            if bayesian_plot:
                print("  Created group effect size plot")
            else:
                print("  Failed to create group effect size plot")
        except Exception as e:
            logger.error(f"Error creating group effect size plot: {e}")
            print(f"  Error creating group effect size plot: {e}")
    else:
        print("  Skipping hierarchical analysis (need >1 protein)")
    
    # Calculate runtime - EXACT SAME AS ORIGINAL
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"\nAnalysis completed in {runtime:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print("Generated files:")
    print(f"- state_distribution.png/svg")
    print(f"- gm3_effect_summary.png/svg")  # Keep original name for compatibility
    if len(all_results) > 1:
        print(f"- group_effect_size.png/svg")
    
    return all_results