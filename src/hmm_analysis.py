#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hidden Markov Model Analysis Module

This module implements state determination and transition analysis for
protein transport between membrane domains with configurable target lipid support.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import logging

from .config import config

logger = logging.getLogger(__name__)


def perform_viterbi_decoding(protein_data, protein_name):
    """
    Implements state determination with configurable target lipid support.
    
    States (dynamically labeled based on target lipid):
    0: Non_TARGET_D - No target lipid binding, D-rich region
    1: Non_TARGET_CS - No target lipid binding, CS-rich region
    2: TARGET_D - Target lipid binding, D-rich region
    3: TARGET_CS - Target lipid binding, CS-rich region
    
    Parameters:
    -----------
    protein_data : dict
        Dictionary of protein data
    protein_name : str
        Name of protein to analyze
        
    Returns:
    --------
    numpy.ndarray
        Array of state assignments
    """
    if not protein_data or protein_name not in protein_data:
        print(f"No data available for {protein_name}. Please load data first.")
        return np.array([])
    
    target_lipid = config.TARGET_LIPID
    state_labels = config.get_state_labels()
    
    print(f"Performing state determination for {protein_name} with target lipid: {target_lipid}")
    
    # Get the protein data
    data = protein_data[protein_name]
    target_lipid_values = data.get('target_lipid_interactions', data.get('gm3_interactions', np.array([])))
    chol_values = data['chol_density']
    sm_values = data.get('sm_density', np.zeros_like(chol_values))
    order_params = data.get('order_parameter', np.full_like(chol_values, 0.5))
    cs_rich_status = data.get('cs_rich', np.zeros_like(target_lipid_values, dtype=bool))
    
    n_samples = len(target_lipid_values)
    states = np.zeros(n_samples, dtype=int)
    
    # Debug output
    print(f"RAW DATA FOR {protein_name}:")
    print(f"  Total frames: {n_samples}")
    print(f"  {target_lipid} range: {np.min(target_lipid_values):.4f} to {np.max(target_lipid_values):.4f}, mean: {np.mean(target_lipid_values):.4f}")
    print(f"  CHOL range: {np.min(chol_values):.4f} to {np.max(chol_values):.4f}, mean: {np.mean(chol_values):.4f}")
    print(f"  SM range: {np.min(sm_values):.4f} to {np.max(sm_values):.4f}, mean: {np.mean(sm_values):.4f}")
    print(f"  Order range: {np.min(order_params):.4f} to {np.max(order_params):.4f}, mean: {np.mean(order_params):.4f}")
    
    # Target lipid binding determination
    has_target_lipid = target_lipid_values > 0.01
    
    # CS-rich determination - vectorize
    is_cs_rich = np.zeros(n_samples, dtype=bool)
    
    if len(cs_rich_status) == n_samples:
        is_cs_rich = cs_rich_status.astype(bool)
    else:
        # Use threshold-based determination
        is_cs_rich = ((chol_values > config.CHOL_DENSITY_THRESHOLD) | 
                     (sm_values > config.SM_DENSITY_THRESHOLD) | 
                     ((chol_values > 0.5) & (sm_values > 0.5)))
    
    # Vectorized state determination
    states = np.zeros(n_samples, dtype=int)
    
    # State 2: TARGET_D (target lipid binding in D-rich region)
    states[(has_target_lipid) & (~is_cs_rich)] = 2
    
    # State 3: TARGET_CS (target lipid binding in CS-rich region)
    states[(has_target_lipid) & (is_cs_rich)] = 3
    
    # State 1: Non_TARGET_CS (No target lipid binding in CS-rich region)
    states[(~has_target_lipid) & (is_cs_rich)] = 1
    
    # State 0: Non_TARGET_D (No target lipid binding in D-rich region) is already 0
    
    # Output state distribution
    state_counts = {}
    for s in range(4):
        state_counts[s] = np.sum(states == s)
        
    total_count = sum(state_counts.values())
    print(f"STATE DISTRIBUTION FOR {protein_name}:")
    for s in range(4):
        count = state_counts[s]
        percentage = (count / total_count) * 100 if total_count > 0 else 0
        print(f"  State {s} ({state_labels[s]}): {count} frames ({percentage:.2f}%)")
    
    # Accuracy check
    target_lipid_count = np.sum(has_target_lipid)
    cs_count = np.sum(is_cs_rich)
    print(f"  Total frames with {target_lipid}: {target_lipid_count} ({target_lipid_count/n_samples*100:.2f}%)")
    print(f"  Total frames in CS-rich: {cs_count} ({cs_count/n_samples*100:.2f}%)")
    print(f"  Total frames: {n_samples}")
    
    return states


def calculate_transition_matrix(states):
    """
    Calculate transition probability matrix from state sequence.
    
    Parameters:
    -----------
    states : numpy.ndarray
        Array of state assignments
        
    Returns:
    --------
    numpy.ndarray
        Transition probability matrix
    """
    state_labels = config.get_state_labels()
    n_states = config.N_STATES
    
    print(f"\n*** CALCULATING TRANSITION MATRIX ***")
    print(f"States: {state_labels}")
    
    if len(states) == 0:
        print("ERROR: No state data provided")
        return np.eye(n_states) * 0.25  # Equal probabilities as fallback
    
    if len(states) < 2:
        print("ERROR: States array too short for transition calculation")
        return np.eye(n_states) * 0.25
    
    # Validate state values
    valid_states = np.all((states >= 0) & (states < n_states))
    if not valid_states:
        invalid_count = np.sum((states < 0) | (states >= n_states))
        print(f"WARNING: {invalid_count} invalid state values found. Valid range: 0-{n_states-1}")
        # Filter invalid states
        valid_mask = (states >= 0) & (states < n_states)
        states = states[valid_mask]
        
        if len(states) < 2:
            print("ERROR: No valid transitions after filtering")
            return np.eye(n_states) * 0.25
    
    print(f"Total states for transition calculation: {len(states)}")
    print(f"State distribution: {np.bincount(states, minlength=n_states)}")
    
    # Create transition count matrix
    transition_counts = np.zeros((n_states, n_states), dtype=int)
    
    # Count state transitions
    for i in range(len(states) - 1):
        from_state = states[i]
        to_state = states[i + 1]
        
        # Additional validation
        if 0 <= from_state < n_states and 0 <= to_state < n_states:
            transition_counts[from_state, to_state] += 1
        else:
            print(f"WARNING: Invalid state transition: {from_state} -> {to_state}")
    
    # Output transition count matrix
    print("Transition counts matrix:")
    for i in range(n_states):
        print(f"  From state {i}: {transition_counts[i]}")
    
    # Calculate transition probability matrix
    transition_matrix = np.zeros((n_states, n_states))
    
    for i in range(n_states):
        row_sum = np.sum(transition_counts[i])
        if row_sum > 0:
            transition_matrix[i] = transition_counts[i] / row_sum
        else:
            # Default probabilities for states with no transitions
            if i == 0:
                transition_matrix[i] = np.array([0.7, 0.15, 0.15, 0.0])
            elif i == 1:
                transition_matrix[i] = np.array([0.15, 0.7, 0.0, 0.15])
            elif i == 2:
                transition_matrix[i] = np.array([0.15, 0.0, 0.7, 0.15])
            elif i == 3:
                transition_matrix[i] = np.array([0.0, 0.15, 0.15, 0.7])
    
    # Validate transition matrix
    for i in range(n_states):
        row_sum = np.sum(transition_matrix[i])
        if not np.isclose(row_sum, 1.0, atol=1e-6):
            if row_sum > 0:
                print(f"WARNING: Row {i} sum is {row_sum}, normalizing...")
                transition_matrix[i] /= row_sum
            else:
                print(f"WARNING: Row {i} is all zeros, setting default probabilities")
                transition_matrix[i] = np.array([0.1, 0.1, 0.1, 0.1])
                transition_matrix[i, i] = 0.7  # Higher probability to self
                transition_matrix[i] /= np.sum(transition_matrix[i])
    
    # Ensure all columns have at least some probability
    for j in range(n_states):
        col_sum = np.sum(transition_matrix[:, j])
        if col_sum < 1e-6:
            print(f"WARNING: Column {j} has very low probability, adjusting...")
            for i in range(n_states):
                transition_matrix[i, j] += 0.05
                transition_matrix[i] /= np.sum(transition_matrix[i])
    
    # Output final transition matrix
    print("Final transition matrix:")
    for i in range(n_states):
        print(f"  From state {i}: {transition_matrix[i]} (sum: {np.sum(transition_matrix[i]):.6f})")
    
    print("*** TRANSITION MATRIX CALCULATION COMPLETE ***\n")
    return transition_matrix


def analyze_target_lipid_transport_effect(transition_matrix, n_steps=20):
    """
    Analyze target lipid transport effect with configurable target lipid.
    
    Parameters:
    -----------
    transition_matrix : numpy.ndarray
        4x4 transition probability matrix
    n_steps : int
        Number of steps for cumulative analysis
        
    Returns:
    --------
    dict
        Dictionary containing transport effect analysis
    """
    target_lipid = config.TARGET_LIPID
    state_labels = config.get_state_labels()
    
    print(f"*** ANALYZING {target_lipid} TRANSPORT EFFECT ***")
    
    # Debug: print input transition matrix
    print("Input transition matrix:")
    if transition_matrix is None:
        print("ERROR: transition_matrix is None!")
        transition_matrix = np.array([
            [0.7, 0.15, 0.15, 0.0],
            [0.15, 0.7, 0.0, 0.15],
            [0.15, 0.0, 0.7, 0.15],
            [0.0, 0.15, 0.15, 0.7]
        ])
    
    # Print the transition matrix
    for i in range(len(transition_matrix)):
        print(f"  From {state_labels[i]}: {transition_matrix[i]}")
    
    # Calculate cumulative effect
    matrix = transition_matrix.copy()
    
    # Track probabilities over time
    initial_state = np.array([1.0, 0.0, 0.0, 0.0])  # Start in Non_TARGET_D
    
    cumulative_probabilities = []
    current_state = initial_state.copy()
    
    for step in range(n_steps):
        current_state = current_state @ matrix
        cumulative_probabilities.append(current_state.copy())
    
    # Calculate transport effects
    transport_effects = {}
    
    # Direct transition effects
    direct_transport = matrix[2, 3] - matrix[0, 1]  # TARGET_D->TARGET_CS vs Non_TARGET_D->Non_TARGET_CS
    transport_effects['direct_transport'] = direct_transport
    
    # Long-term equilibrium
    final_probs = cumulative_probabilities[-1]
    transport_effects['equilibrium_target_cs'] = final_probs[3]
    transport_effects['equilibrium_non_target_cs'] = final_probs[1]
    
    print(f"Transport effect analysis for {target_lipid}:")
    print(f"  Direct transport enhancement: {direct_transport:.4f}")
    print(f"  Long-term {target_lipid}_CS probability: {final_probs[3]:.4f}")
    print(f"  Long-term Non_{target_lipid}_CS probability: {final_probs[1]:.4f}")
    
    return {
        'transport_effects': transport_effects,
        'cumulative_probabilities': cumulative_probabilities,
        'transition_matrix': transition_matrix
    }