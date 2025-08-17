#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Loading Module for GM3-Cholesterol Rich Domain Transport Analysis

This module handles MD trajectory loading, protein identification, and 
lipid/membrane component selection using MDAnalysis.
"""

import os
import numpy as np
import MDAnalysis as mda
import logging

logger = logging.getLogger(__name__)


def load_universe(logger=None):
    """
    Load trajectory using MDAnalysis with configurable file paths.
    
    Returns:
    --------
    MDAnalysis.Universe or None
        Universe object if successful, None otherwise
    """
    from .config import config
    
    if logger is None:
        logger = logging.getLogger(__name__)
        
    try:
        topology_file = config.TOPOLOGY_FILE
        trajectory_file = config.TRAJECTORY_FILE
        
        print(f"Loading trajectory from:")
        print(f"  Topology: {topology_file}")
        print(f"  Trajectory: {trajectory_file}")
        
        # Check if trajectory files exist
        if os.path.exists(topology_file) and os.path.exists(trajectory_file):
            u = mda.Universe(topology_file, trajectory_file)
            print("Trajectory loaded successfully.")
            print(f"  Number of atoms: {len(u.atoms)}")
            print(f"  Number of frames: {len(u.trajectory)}")
            return u
        else:
            missing_files = []
            if not os.path.exists(topology_file):
                missing_files.append(topology_file)
            if not os.path.exists(trajectory_file):
                missing_files.append(trajectory_file)
            
            error_msg = f"Trajectory files not found: {missing_files}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            print("Please check file paths or use --topology and --trajectory arguments")
            return None
            
    except Exception as e:
        logger.error(f"Error loading trajectory: {e}")
        print(f"Error loading trajectory: {e}")
        return None


def identify_lipid_leaflets(u, logger=None):
    """
    Identify lipid leaflets in the membrane.
    
    Parameters:
    -----------
    u : MDAnalysis.Universe
        Universe object
        
    Returns:
    --------
    MDAnalysis.AtomGroup or None
        Leaflet group if successful, None otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
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


def identify_proteins(u, logger=None):
    """
    Identify proteins in the trajectory.
    
    Parameters:
    -----------
    u : MDAnalysis.Universe
        Universe object
        
    Returns:
    --------
    dict
        Dictionary of protein selections by name
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    proteins = {}
    try:
        protein_residues = u.select_atoms("protein")
        if len(protein_residues) == 0:
            # Try different selection if standard "protein" doesn't work
            protein_residues = u.select_atoms("resname PROT")
        
        if len(protein_residues) == 0:
            logger.warning("No protein residues found in trajectory")
            return {}
        
        # Group by segid or similar to identify individual proteins
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
    """
    Select all detected lipids from the leaflet.
    
    Parameters:
    -----------
    lipid_types : list
        List of lipid residue names (auto-detected from trajectory)
    leaflet : MDAnalysis.AtomGroup
        Leaflet to select from
        
    Returns:
    --------
    dict
        Dictionary of lipid selections by type
    """
    selections = {}
    
    if lipid_types is None or len(lipid_types) == 0:
        print("Warning: No lipid types provided for selection")
        return selections
    
    print(f"Selecting lipids from leaflet: {lipid_types}")
    
    # Select each detected lipid type
    for resname in lipid_types:
        try:
            selection = leaflet.select_atoms(f"resname {resname}")
            if len(selection) > 0:
                selections[resname] = selection
                print(f"  {resname}: {len(selection)} atoms")
            else:
                print(f"  {resname}: 0 atoms (skipping)")
                
        except Exception as e:
            print(f"Could not select lipid type {resname}: {e}")
            # Create an empty selection as a fallback
            selections[resname] = mda.AtomGroup([], leaflet.universe)
    
    # Report final selections
    print(f"Successfully selected {len(selections)} lipid types from leaflet")
    return selections


def auto_detect_system_components(universe):
    """
    Auto-detect all system components from the universe.
    
    Parameters:
    -----------
    universe : MDAnalysis.Universe
        Loaded trajectory universe
        
    Returns:
    --------
    dict
        Dictionary containing detected components
    """
    from .config import config
    
    print("=== Auto-detecting System Components ===")
    
    # Initialize configuration from trajectory
    success = config.initialize_from_trajectory(universe)
    
    if not success:
        print("Failed to auto-detect system components")
        return {}
    
    return {
        'lipid_types': config.LIPID_TYPES,
        'protein_names': config.PROTEIN_NAMES,
        'all_residues': config.ALL_RESIDUE_TYPES,
        'target_lipid': config.TARGET_LIPID
    }