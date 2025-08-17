#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration Module for GM3-Cholesterol Rich Domain Transport Analysis

This module contains all configurable parameters for the analysis pipeline,
including target lipid selection, analysis parameters, and output settings.
"""

import os
from typing import List, Dict, Any


class AnalysisConfig:
    """Configuration class for the domain transport analysis."""
    
    # =============================================================================
    # TARGET LIPID CONFIGURATION
    # =============================================================================
    # Main target lipid for transport analysis (can be GM3, DPG3, or other gangliosides)
    TARGET_LIPID: str = "DPG3"  # Default to DPG3 (GM3 ganglioside)
    
    # Alternative names for the target lipid (case-insensitive matching)
    TARGET_LIPID_ALIASES: List[str] = ["GM3", "gm3", "DPG3", "dpg3"]
    
    # Auto-detected components (populated from PSF/trajectory)
    LIPID_TYPES: List[str] = None  # Will be auto-detected from trajectory
    PROTEIN_NAMES: List[str] = None  # Will be auto-detected from trajectory
    ALL_RESIDUE_TYPES: List[str] = None  # All residue types in system
    
    # Components to exclude from lipid analysis
    EXCLUDE_FROM_LIPIDS: List[str] = ["TIP3", "POT", "CLA", "SOD", "WAT", "Na+", "Cl-"]
    
    # =============================================================================
    # TRAJECTORY AND SYSTEM PARAMETERS
    # =============================================================================
    # Trajectory files (can be overridden via command line or config file)
    TOPOLOGY_FILE: str = "test_system.psf"
    TRAJECTORY_FILE: str = "test_trajectory.xtc"
    
    # Analysis frame range
    START_FRAME: int = 0
    STOP_FRAME: int = 40
    FRAME_STEP: int = 1
    
    # =============================================================================
    # SPATIAL ANALYSIS PARAMETERS
    # =============================================================================
    # Grid and spatial parameters
    MAX_GRID_SIZE: int = 100
    GRID_SPACING: float = 1.0
    RADIUS: float = 10.0
    
    # Protein structure parameters
    HELIX_RADIUS: float = 3.0
    INTERFACE_WIDTH: float = 2.0
    
    # Interaction radii
    TARGET_LIPID_INTERACTION_RADIUS: float = 10.0  # Radius for target lipid interactions
    CHOL_SHELL_RADIUS: float = 12.0
    
    # =============================================================================
    # DOMAIN CLASSIFICATION PARAMETERS
    # =============================================================================
    # Density thresholds for domain classification
    CHOL_DENSITY_THRESHOLD: float = 0.8
    SM_DENSITY_THRESHOLD: float = 0.55
    
    # Hidden Markov Model parameters
    N_STATES: int = 4  # Four states: Non_Target_D, Non_Target_CS, Target_D, Target_CS
    
    # =============================================================================
    # COMPUTATIONAL PARAMETERS
    # =============================================================================
    # Parallel processing
    BATCH_SIZE: int = 50
    DEFAULT_N_JOBS: int = -1  # Use all available CPUs
    
    # =============================================================================
    # OUTPUT CONFIGURATION
    # =============================================================================
    # Figure output formats
    OUTPUT_FORMATS: List[str] = ["png", "svg"]
    FIGURE_DPI: int = 300
    
    # Output file names (will be formatted with target lipid name)
    OUTPUT_FILES: Dict[str, str] = {
        "state_distribution": "state_distribution",
        "effect_summary": "target_lipid_effect_summary", 
        "group_effect": "group_effect_size"
    }
    
    @classmethod
    def get_state_labels(cls) -> List[str]:
        """Get state labels with current target lipid."""
        target = cls.TARGET_LIPID.upper()
        return [
            f"Non_{target}_D",    # State 0: No target lipid, disordered domain
            f"Non_{target}_CS",   # State 1: No target lipid, cholesterol-rich domain  
            f"{target}_D",        # State 2: Target lipid present, disordered domain
            f"{target}_CS"        # State 3: Target lipid present, cholesterol-rich domain
        ]
    
    @classmethod
    def get_target_lipid_aliases(cls) -> List[str]:
        """Get all possible aliases for target lipid (case-insensitive)."""
        aliases = [cls.TARGET_LIPID.upper(), cls.TARGET_LIPID.lower()]
        for alias in cls.TARGET_LIPID_ALIASES:
            aliases.extend([alias.upper(), alias.lower()])
        return list(set(aliases))
    
    @classmethod
    def initialize_from_trajectory(cls, universe) -> bool:
        """
        Initialize configuration from trajectory data.
        
        Parameters:
        -----------
        universe : MDAnalysis.Universe
            Loaded trajectory universe
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            print("Auto-detecting system components from trajectory...")
            
            # Detect all residue types
            all_residues = universe.atoms
            cls.ALL_RESIDUE_TYPES = sorted(list(set(all_residues.resnames)))
            print(f"Found {len(cls.ALL_RESIDUE_TYPES)} residue types: {cls.ALL_RESIDUE_TYPES}")
            
            # Detect proteins
            try:
                protein_atoms = universe.select_atoms("protein")
                if len(protein_atoms) == 0:
                    # Try alternative protein selections
                    protein_atoms = universe.select_atoms("resname PROT")
                
                if len(protein_atoms) > 0:
                    protein_segids = sorted(list(set(protein_atoms.segids)))
                    cls.PROTEIN_NAMES = [f"Protein_{segid}" for segid in protein_segids]
                    print(f"Detected {len(cls.PROTEIN_NAMES)} proteins: {cls.PROTEIN_NAMES}")
                else:
                    cls.PROTEIN_NAMES = []
                    print("No proteins detected in trajectory")
            except:
                cls.PROTEIN_NAMES = []
                print("Could not detect proteins")
            
            # Detect lipids (exclude water, ions, proteins)
            exclude_set = set(cls.EXCLUDE_FROM_LIPIDS)
            
            # Get non-protein, non-water, non-ion residues
            try:
                non_protein = universe.select_atoms("not protein")
                lipid_candidates = []
                
                for resname in set(non_protein.resnames):
                    if resname not in exclude_set:
                        # Additional check: avoid common water/ion names
                        if not any(x in resname.upper() for x in ['TIP', 'WAT', 'HOH', 'SOL']):
                            lipid_candidates.append(resname)
                
                cls.LIPID_TYPES = sorted(lipid_candidates)
                print(f"Detected {len(cls.LIPID_TYPES)} lipid types: {cls.LIPID_TYPES}")
                
            except Exception as e:
                print(f"Error detecting lipids: {e}")
                cls.LIPID_TYPES = []
            
            # Validate target lipid
            cls._validate_target_lipid()
            
            return True
            
        except Exception as e:
            print(f"Error initializing from trajectory: {e}")
            return False
    
    @classmethod
    def _validate_target_lipid(cls) -> None:
        """Validate that target lipid exists in the trajectory."""
        if cls.LIPID_TYPES is None:
            return
            
        # Check if target lipid or its aliases exist
        available_lipids = set(cls.LIPID_TYPES)
        target_aliases = set(cls.get_target_lipid_aliases())
        
        found_lipids = available_lipids.intersection(target_aliases)
        
        if found_lipids:
            # Use the first found alias as the canonical target lipid
            cls.TARGET_LIPID = sorted(found_lipids)[0]
            print(f"Target lipid '{cls.TARGET_LIPID}' found in trajectory")
        else:
            print(f"WARNING: Target lipid '{cls.TARGET_LIPID}' and aliases {cls.TARGET_LIPID_ALIASES} not found")
            print(f"Available lipids: {cls.LIPID_TYPES}")
            print("Consider updating TARGET_LIPID or TARGET_LIPID_ALIASES")
    
    @classmethod
    def set_target_lipid(cls, target_lipid: str) -> None:
        """Set the target lipid for analysis."""
        cls.TARGET_LIPID = target_lipid.upper()
        if cls.LIPID_TYPES is not None:
            cls._validate_target_lipid()
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration parameters."""
        try:
            # Check that trajectory files exist
            if not (os.path.exists(cls.TOPOLOGY_FILE) and os.path.exists(cls.TRAJECTORY_FILE)):
                print(f"Warning: Trajectory files {cls.TOPOLOGY_FILE} or {cls.TRAJECTORY_FILE} not found")
                return False
                
            # Check frame parameters
            if cls.START_FRAME >= cls.STOP_FRAME:
                print("Error: START_FRAME must be less than STOP_FRAME")
                return False
                
            if cls.FRAME_STEP <= 0:
                print("Error: FRAME_STEP must be positive")
                return False
                
            # Check spatial parameters
            if cls.RADIUS <= 0 or cls.GRID_SPACING <= 0:
                print("Error: Spatial parameters must be positive")
                return False
                
            return True
            
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False
    
    @classmethod
    def print_config(cls) -> None:
        """Print current configuration."""
        print("=== Analysis Configuration ===")
        print(f"Topology File: {cls.TOPOLOGY_FILE}")
        print(f"Trajectory File: {cls.TRAJECTORY_FILE}")
        print(f"Target Lipid: {cls.TARGET_LIPID}")
        print(f"Target Lipid Aliases: {cls.TARGET_LIPID_ALIASES}")
        print(f"State Labels: {cls.get_state_labels()}")
        
        if cls.LIPID_TYPES is not None:
            print(f"Detected Lipid Types ({len(cls.LIPID_TYPES)}): {cls.LIPID_TYPES}")
        else:
            print("Lipid Types: Not yet detected from trajectory")
            
        if cls.PROTEIN_NAMES is not None:
            print(f"Detected Proteins ({len(cls.PROTEIN_NAMES)}): {cls.PROTEIN_NAMES}")
        else:
            print("Proteins: Not yet detected from trajectory")
            
        if cls.ALL_RESIDUE_TYPES is not None:
            print(f"All Residue Types ({len(cls.ALL_RESIDUE_TYPES)}): {cls.ALL_RESIDUE_TYPES}")
            
        print(f"Frame Range: {cls.START_FRAME}-{cls.STOP_FRAME} (step {cls.FRAME_STEP})")
        print(f"Grid Spacing: {cls.GRID_SPACING}")
        print(f"Interaction Radius: {cls.TARGET_LIPID_INTERACTION_RADIUS}")
        print("==============================")


# Global configuration instance
config = AnalysisConfig()


def load_config_from_file(config_file: str) -> bool:
    """
    Load configuration from a JSON or YAML file.
    
    Parameters:
    -----------
    config_file : str
        Path to configuration file
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        import json
        
        if not os.path.exists(config_file):
            print(f"Config file {config_file} not found")
            return False
            
        with open(config_file, 'r') as f:
            if config_file.endswith('.json'):
                config_data = json.load(f)
            else:
                print("Only JSON config files are currently supported")
                return False
        
        # Update configuration
        for key, value in config_data.items():
            if hasattr(AnalysisConfig, key):
                setattr(AnalysisConfig, key, value)
            else:
                print(f"Warning: Unknown config parameter {key}")
        
        return True
        
    except Exception as e:
        print(f"Error loading config file: {e}")
        return False


def create_example_config(output_file: str = "config_example.json") -> None:
    """Create an example configuration file."""
    example_config = {
        "TARGET_LIPID": "DPG3",
        "TARGET_LIPID_ALIASES": ["GM3", "gm3", "DPG3", "dpg3"],
        "LIPID_TYPES": ["DPPC", "DPSM", "DPGS", "CHOL"],
        "START_FRAME": 20000,
        "STOP_FRAME": 80000,
        "FRAME_STEP": 10,
        "TARGET_LIPID_INTERACTION_RADIUS": 10.0,
        "CHOL_DENSITY_THRESHOLD": 0.8,
        "SM_DENSITY_THRESHOLD": 0.55
    }
    
    import json
    with open(output_file, 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print(f"Example configuration saved to {output_file}")