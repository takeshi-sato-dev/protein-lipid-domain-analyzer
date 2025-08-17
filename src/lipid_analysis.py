#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lipid Analysis Module for GM3-Cholesterol Rich Domain Transport Analysis

This module handles lipid order parameter calculations, domain mapping,
and target lipid interaction analysis with configurable target lipid support.
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
import logging

from .config import config

logger = logging.getLogger(__name__)


def calculate_order_parameter(residue, chainA='name ??A', chainB='name ??B'):
    """
    Calculate order parameter for a lipid residue.
    
    Parameters:
    -----------
    residue : MDAnalysis.AtomGroup
        Residue to calculate order parameter for
    chainA : str
        Selection string for chain A
    chainB : str
        Selection string for chain B
        
    Returns:
    --------
    float
        Order parameter value (or NaN if calculation is not possible)
    """
    try:
        chain_atomsA = residue.atoms.select_atoms(chainA)
        chain_atomsB = residue.atoms.select_atoms(chainB)
        
        if len(chain_atomsA) < 2 or len(chain_atomsB) < 2:
            return np.nan

        def get_chain_vector(chain_atoms):
            vectors = np.diff(chain_atoms.positions, axis=0)
            norms = np.linalg.norm(vectors, axis=1)
            norms[norms == 0] = np.finfo(float).eps
            unit_vectors = vectors / norms[:, np.newaxis]
            return unit_vectors[:, 2]
        
        cos_thetasA = get_chain_vector(chain_atomsA)
        cos_thetasB = get_chain_vector(chain_atomsB)

        S_CD_A = (3 * np.nanmean(cos_thetasA ** 2) - 1) / 2
        S_CD_B = (3 * np.nanmean(cos_thetasB ** 2) - 1) / 2

        S_CD = (S_CD_A + S_CD_B) / 2
        return S_CD if np.isfinite(S_CD) else np.nan
        
    except Exception as e:
        return np.nan


def calculate_op_kde(lipid_data, exclude_points=None):
    """
    Calculate kernel density estimation for order parameter data.
    
    Parameters:
    -----------
    lipid_data : pandas.DataFrame
        Lipid data with positions and order parameters
    exclude_points : numpy.ndarray, optional
        Points to exclude from the KDE
        
    Returns:
    --------
    scipy.stats.gaussian_kde
        KDE object for the order parameter data
    """
    try:
        # Use all lipids (including target lipid) for KDE calculation
        positions = lipid_data[['x', 'y']].values
        op_values = lipid_data['S_CD'].values
        
        # Filter valid data
        valid_mask = np.isfinite(op_values)
        positions = positions[valid_mask]
        op_values = op_values[valid_mask]
        
        if exclude_points is not None:
            mask = np.ones(len(positions), dtype=bool)
            distances = distance.cdist(positions, exclude_points)
            mask &= np.all(distances > config.GRID_SPACING, axis=1)
            positions = positions[mask]
            op_values = op_values[mask]
        
        if len(positions) < 2:
            return None
            
        if not np.all(np.isfinite(positions)):
            finite_mask = np.all(np.isfinite(positions), axis=1)
            positions = positions[finite_mask]
            op_values = op_values[finite_mask]
            
            if len(positions) < 2:
                return None
        
        try:
            weights = op_values - np.min(op_values) + 1e-10
            if not np.all(np.isfinite(weights)):
                return None
                
            return gaussian_kde(positions.T, weights=weights, bw_method=0.15)
        except Exception as e:
            print(f"Error in KDE calculation: {e}")
            return None
            
    except Exception as e:
        print(f"Error in calculate_op_kde: {e}")
        return None


def find_target_lipid_in_selections(selections):
    """
    Find target lipid in the available selections using configured aliases.
    
    Parameters:
    -----------
    selections : dict
        Dictionary of lipid selections
        
    Returns:
    --------
    str or None
        Name of target lipid if found, None otherwise
    """
    target_aliases = config.get_target_lipid_aliases()
    
    for alias in target_aliases:
        if alias in selections:
            print(f"Found target lipid as: {alias}")
            return alias
    
    print(f"Target lipid not found. Searched for: {target_aliases}")
    print(f"Available selections: {list(selections.keys())}")
    return None


def calculate_domain_info(density, lipid_data, lipid_positions, dimensions, x_grid, y_grid):
    """
    Calculate membrane domain characteristics with configurable target lipid support.
    Identifies CS rich regions (Core + surrounding) and D rich regions.
    
    Parameters:
    -----------
    density : numpy.ndarray
        Overall density map
    lipid_data : pandas.DataFrame
        Lipid position and property data
    lipid_positions : dict
        Dictionary containing position information for each lipid type
    dimensions : numpy.ndarray
        System dimensions
    x_grid, y_grid : numpy.ndarray
        Grid coordinates
        
    Returns:
    --------
    dict
        Dictionary containing domain information, thresholds, and statistics
    """
    try:
        # Prepare grid points
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        
        # 1. Calculate cholesterol density
        if 'CHOL' in lipid_positions and len(lipid_positions['CHOL']) > 0:
            kde = gaussian_kde(lipid_positions['CHOL'].T)
            chol_density = kde(positions).reshape(x_grid.shape)
        else:
            chol_density = np.zeros_like(density)
        
        # 2. Calculate sphingomyelin density
        if 'DPSM' in lipid_positions and len(lipid_positions['DPSM']) > 0:
            kde = gaussian_kde(lipid_positions['DPSM'].T)
            sm_density = kde(positions).reshape(x_grid.shape)
        else:
            sm_density = np.zeros_like(density)
        
        # 3. Calculate target lipid density (configurable)
        target_lipid_key = find_target_lipid_in_selections(lipid_positions)
        has_target_lipid = False
        
        if target_lipid_key and len(lipid_positions[target_lipid_key]) > 0:
            kde = gaussian_kde(lipid_positions[target_lipid_key].T)
            target_lipid_density = kde(positions).reshape(x_grid.shape)
            has_target_lipid = True
            print(f"Using {target_lipid_key} as target lipid for domain analysis")
        else:
            target_lipid_density = np.zeros_like(density)
            print("No target lipid found for domain analysis")
        
        # 4. Grid the order parameters
        order_params = np.zeros_like(density)
        for idx, row in lipid_data.iterrows():
            if not np.isnan(row['S_CD']):
                x_idx = int((row['x'] % dimensions[0]) * order_params.shape[0] / dimensions[0])
                y_idx = int((row['y'] % dimensions[1]) * order_params.shape[1] / dimensions[1])
                if 0 <= x_idx < order_params.shape[0] and 0 <= y_idx < order_params.shape[1]:
                    order_params[x_idx, y_idx] = row['S_CD']
        
        # 5. Normalize maps
        def safe_normalize(arr):
            arr_min = np.min(arr)
            arr_max = np.max(arr)
            arr_range = arr_max - arr_min
            if arr_range < 1e-10:
                return np.full_like(arr, 0.5)
            else:
                return (arr - arr_min) / arr_range
        
        density_norm = safe_normalize(density)
        order_norm = safe_normalize(order_params)
        chol_norm = safe_normalize(chol_density)
        sm_norm = safe_normalize(sm_density)
        
        # 6. Set weights based on target lipid presence
        if has_target_lipid:
            # When target lipid exists, increase weights for cholesterol and sphingomyelin
            weights = {'density': 0.2, 'order': 0.25, 'chol': 0.3, 'sm': 0.25}
        else:
            # Weights when target lipid is not present
            weights = {'density': 0.25, 'order': 0.3, 'chol': 0.25, 'sm': 0.2}
        
        # 7. Calculate integrated score
        cs_rich_score = (
            weights['density'] * density_norm + 
            weights['order'] * order_norm + 
            weights['chol'] * chol_norm + 
            weights['sm'] * sm_norm
        )
        
        # 8. Spatial smoothing
        cs_rich_score_smooth = gaussian_filter(cs_rich_score, sigma=1.5)
        
        # 9. Determine CS rich region threshold
        core_threshold = np.mean(cs_rich_score_smooth) + 1.0 * np.std(cs_rich_score_smooth)
        cs_threshold = np.mean(cs_rich_score_smooth) + 0.6 * np.std(cs_rich_score_smooth)
        
        # 10. Identify domains
        core_cs_rich = cs_rich_score_smooth > core_threshold
        cs_rich = cs_rich_score_smooth > cs_threshold
        d_rich = ~cs_rich
        
        # 11. Calculate statistics for each domain
        domain_stats = {
            'area_fraction_core_cs': np.sum(core_cs_rich) / core_cs_rich.size,
            'area_fraction_cs': np.sum(cs_rich) / cs_rich.size,
            'area_fraction_d': np.sum(d_rich) / d_rich.size,
            
            'mean_order_core_cs': np.mean(order_params[core_cs_rich]) if np.any(core_cs_rich) else 0,
            'mean_order_cs': np.mean(order_params[cs_rich]) if np.any(cs_rich) else 0,
            'mean_order_d': np.mean(order_params[d_rich]) if np.any(d_rich) else 0,
            
            'mean_chol_core_cs': np.mean(chol_density[core_cs_rich]) if np.any(core_cs_rich) else 0,
            'mean_chol_cs': np.mean(chol_density[cs_rich]) if np.any(cs_rich) else 0,
            'mean_chol_d': np.mean(chol_density[d_rich]) if np.any(d_rich) else 0,
            
            'target_lipid_key': target_lipid_key,
            'has_target_lipid': has_target_lipid
        }
        
        return {
            'cs_rich_map': cs_rich,
            'core_cs_rich_map': core_cs_rich,
            'd_rich_map': d_rich,
            'cs_rich_score': cs_rich_score_smooth,
            'chol_density': chol_density,
            'sm_density': sm_density,
            'target_lipid_density': target_lipid_density,
            'order_params': order_params,
            'domain_stats': domain_stats,
            'thresholds': {
                'core_threshold': core_threshold,
                'cs_threshold': cs_threshold
            }
        }
        
    except Exception as e:
        logger.error(f"Error in calculate_domain_info: {e}")
        print(f"Error in calculate_domain_info: {e}")
        return None


def analyze_lipid_distribution(protein_com, selections, box_dimensions):
    """
    Analyze lipid distribution around protein center of mass with configurable target lipid.
    
    Parameters:
    -----------
    protein_com : numpy.ndarray
        Protein center of mass coordinates
    selections : dict
        Dictionary of lipid selections
    box_dimensions : numpy.ndarray
        System box dimensions
        
    Returns:
    --------
    dict
        Dictionary containing lipid analysis results
    """
    try:
        # Find target lipid
        target_lipid_key = find_target_lipid_in_selections(selections)
        
        results = {
            'target_lipid_key': target_lipid_key,
            'target_lipid_count': 0,
            'target_lipid_distances': [],
            'chol_count': 0,
            'chol_distances': []
        }
        
        # Analyze target lipid distribution
        if target_lipid_key and target_lipid_key in selections:
            target_selection = selections[target_lipid_key]
            if len(target_selection) > 0:
                # Calculate distances to target lipid
                target_positions = target_selection.positions[:, :2]  # xy only
                distances = np.linalg.norm(target_positions - protein_com[:2], axis=1)
                
                # Count target lipids within interaction radius
                close_target = distances <= config.TARGET_LIPID_INTERACTION_RADIUS
                results['target_lipid_count'] = np.sum(close_target)
                results['target_lipid_distances'] = distances[close_target].tolist()
        
        # Analyze cholesterol distribution
        if 'CHOL' in selections:
            chol_selection = selections['CHOL']
            if len(chol_selection) > 0:
                chol_positions = chol_selection.positions[:, :2]  # xy only
                distances = np.linalg.norm(chol_positions - protein_com[:2], axis=1)
                
                # Count cholesterol within shell radius
                close_chol = distances <= config.CHOL_SHELL_RADIUS
                results['chol_count'] = np.sum(close_chol)
                results['chol_distances'] = distances[close_chol].tolist()
        
        return results
        
    except Exception as e:
        logger.error(f"Error in analyze_lipid_distribution: {e}")
        return {'target_lipid_key': None, 'target_lipid_count': 0, 'target_lipid_distances': [], 
                'chol_count': 0, 'chol_distances': []}