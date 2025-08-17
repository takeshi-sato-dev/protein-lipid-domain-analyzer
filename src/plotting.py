#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting Module for GM3-Cholesterol Rich Domain Transport Analysis

This module handles all figure generation for the analysis pipeline,
with configurable target lipid support.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import logging

matplotlib.use('Agg')  # Use non-interactive backend

from .config import config

logger = logging.getLogger(__name__)


def create_state_distribution_plot(protein_data, plot_proteins, output_dir):
    """
    Create state distribution plot with configurable target lipid labels.
    
    Parameters:
    -----------
    protein_data : dict
        Dictionary of protein data
    plot_proteins : list
        List of proteins to plot
    output_dir : str
        Directory to save plots
        
    Returns:
    --------
    dict
        Dictionary containing plot paths
    """
    try:
        print("Creating state distribution plot")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        state_labels = config.get_state_labels()
        target_lipid = config.TARGET_LIPID
        
        # Filter proteins with valid data
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
        
        # Create plot
        n_proteins = len(valid_proteins)
        x_positions = np.arange(n_proteins)
        width = 0.2
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # Create bars for each state
        for state in range(4):
            percentages = [valid_data[protein]['percentages'][state] for protein in valid_proteins]
            plt.bar(x_positions + state * width, percentages, width, 
                   label=state_labels[state], color=colors[state], alpha=0.8)
        
        # Customize plot
        plt.xlabel('Protein', fontsize=12, fontweight='bold')
        plt.ylabel('Percentage of Frames (%)', fontsize=12, fontweight='bold')
        plt.title(f'Protein State Distribution - {target_lipid} Analysis', fontsize=14, fontweight='bold')
        plt.xticks(x_positions + 1.5 * width, valid_proteins, rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save plots
        plot_path_png = os.path.join(output_dir, 'state_distribution.png')
        plot_path_svg = os.path.join(output_dir, 'state_distribution.svg')
        plt.savefig(plot_path_png, dpi=config.FIGURE_DPI, bbox_inches='tight')
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
        print(f"Error creating state distribution plot: {e}")
        return {}


def create_target_lipid_effect_summary_plot(protein_data, plot_proteins, output_dir):
    """
    Create target lipid effect summary plot with configurable target lipid.
    
    Parameters:
    -----------
    protein_data : dict
        Dictionary of protein data
    plot_proteins : list
        List of proteins to plot
    output_dir : str
        Directory to save plots
        
    Returns:
    --------
    dict
        Dictionary containing plot paths
    """
    try:
        print("Creating target lipid effect summary plot")
        
        target_lipid = config.TARGET_LIPID
        state_labels = config.get_state_labels()
        
        # Collect effect data
        transport_effects = []
        protein_names = []
        
        for protein_name in plot_proteins:
            if protein_name in protein_data:
                analysis = protein_data[protein_name].get('analysis', {})
                transport_effect = analysis.get('transport_effect', None)
                
                if transport_effect is not None:
                    transport_effects.append(transport_effect)
                    protein_names.append(protein_name)
                    print(f"  {protein_name}: Transport effect = {transport_effect:.4f}")
        
        if len(transport_effects) == 0:
            print("No meaningful effect data found for any protein, creating default summary plot")
            
            # Create a simple informational plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'No {target_lipid} Transport Effects Detected\n\n'
                              f'This indicates that {target_lipid} does not significantly\n'
                              f'enhance protein transport to cholesterol-rich domains\n'
                              f'in the analyzed trajectory.',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{target_lipid} Transport Effect Summary', fontsize=16, fontweight='bold')
            
        else:
            # Create actual effect plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Transport effects by protein
            colors = plt.cm.viridis(np.linspace(0, 1, len(protein_names)))
            bars = ax1.bar(range(len(protein_names)), transport_effects, color=colors, alpha=0.7)
            ax1.set_xlabel('Protein', fontweight='bold')
            ax1.set_ylabel(f'{target_lipid} Transport Effect', fontweight='bold')
            ax1.set_title(f'{target_lipid}-Mediated Transport Effects', fontweight='bold')
            ax1.set_xticks(range(len(protein_names)))
            ax1.set_xticklabels(protein_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # Add value labels on bars
            for bar, effect in zip(bars, transport_effects):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{effect:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Plot 2: Effect distribution
            if len(transport_effects) > 1:
                ax2.hist(transport_effects, bins=min(len(transport_effects), 10), 
                        alpha=0.7, color='skyblue', edgecolor='black')
                ax2.set_xlabel(f'{target_lipid} Transport Effect', fontweight='bold')
                ax2.set_ylabel('Frequency', fontweight='bold')
                ax2.set_title('Distribution of Transport Effects', fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.axvline(x=np.mean(transport_effects), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(transport_effects):.3f}')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'Single Protein\nAnalysis', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes, fontsize=14)
                ax2.set_xticks([])
                ax2.set_yticks([])
        
        plt.tight_layout()
        
        # Save plots
        plot_path_png = os.path.join(output_dir, f'{target_lipid.lower()}_effect_summary.png')
        plot_path_svg = os.path.join(output_dir, f'{target_lipid.lower()}_effect_summary.svg')
        plt.savefig(plot_path_png, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.savefig(plot_path_svg, format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"{target_lipid} effect summary plot saved:")
        print(f"  PNG: {plot_path_png}")
        print(f"  SVG: {plot_path_svg}")
        
        return {
            'png': plot_path_png,
            'svg': plot_path_svg,
            'effects': transport_effects,
            'proteins': protein_names
        }
        
    except Exception as e:
        logger.error(f"Error creating {target_lipid} effect summary plot: {e}")
        print(f"Error creating {target_lipid} effect summary plot: {e}")
        return {}


def perform_hierarchical_bayesian_analysis(protein_data, output_dir):
    """
    Perform hierarchical Bayesian analysis and create group effect plot.
    
    Parameters:
    -----------
    protein_data : dict
        Dictionary of protein data
    output_dir : str
        Directory to save plots
        
    Returns:
    --------
    dict
        Dictionary containing analysis results and plot paths
    """
    try:
        import pymc as pm
        import arviz as az
        
        target_lipid = config.TARGET_LIPID
        print(f"Performing hierarchical Bayesian analysis for {target_lipid} effects")
        
        # Extract transport effects
        effects = []
        protein_names = []
        
        for protein_name, data in protein_data.items():
            analysis = data.get('analysis', {})
            transport_effect = analysis.get('transport_effect', None)
            
            if transport_effect is not None and np.isfinite(transport_effect):
                effects.append(transport_effect)
                protein_names.append(protein_name)
        
        if len(effects) < 2:
            print(f"Insufficient data for Bayesian analysis (need ≥2 proteins, have {len(effects)})")
            return {}
        
        print(f"Analyzing {len(effects)} proteins for group-level effects")
        
        # Bayesian hierarchical model
        with pm.Model() as model:
            # Group-level parameters
            mu_group = pm.Normal('mu_group', mu=0, sigma=1)
            sigma_group = pm.HalfNormal('sigma_group', sigma=1)
            
            # Individual protein effects
            theta = pm.Normal('theta', mu=mu_group, sigma=sigma_group, shape=len(effects))
            
            # Likelihood
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.1)
            obs = pm.Normal('obs', mu=theta, sigma=sigma_obs, observed=effects)
            
            # Sample
            trace = pm.sample(2000, tune=1000, chains=2, target_accept=0.95, 
                             random_seed=42, progressbar=True)
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Group effect posterior
        az.plot_posterior(trace, var_names=['mu_group'], ax=axes[0,0])
        axes[0,0].set_title(f'Group-Level {target_lipid} Effect', fontweight='bold')
        
        # Plot 2: Individual effects
        az.plot_forest(trace, var_names=['theta'], ax=axes[0,1])
        axes[0,1].set_title('Individual Protein Effects', fontweight='bold')
        axes[0,1].set_yticklabels(protein_names)
        
        # Plot 3: Trace plot
        az.plot_trace(trace, var_names=['mu_group'], ax=axes[1,0])
        axes[1,0][0].set_title('Group Effect Trace', fontweight='bold')
        
        # Plot 4: Effect size distribution
        group_effects = trace.posterior['mu_group'].values.flatten()
        axes[1,1].hist(group_effects, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1,1].axvline(np.mean(group_effects), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(group_effects):.3f}')
        axes[1,1].set_xlabel(f'Group {target_lipid} Effect')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Group Effect Distribution', fontweight='bold')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path_png = os.path.join(output_dir, "group_effect_size.png")
        plot_path_svg = os.path.join(output_dir, "group_effect_size.svg")
        plt.savefig(plot_path_png, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.savefig(plot_path_svg, format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"Hierarchical Bayesian analysis completed:")
        print(f"  PNG: {plot_path_png}")
        print(f"  SVG: {plot_path_svg}")
        print(f"  Group effect mean: {np.mean(group_effects):.4f}")
        print(f"  Group effect std: {np.std(group_effects):.4f}")
        
        return {
            'png': plot_path_png,
            'svg': plot_path_svg,
            'trace': trace,
            'group_effect_mean': np.mean(group_effects),
            'group_effect_std': np.std(group_effects),
            'individual_effects': effects,
            'protein_names': protein_names
        }
        
    except Exception as e:
        logger.error(f"Error in hierarchical Bayesian analysis: {e}")
        print(f"Error in hierarchical Bayesian analysis: {e}")
        return {}