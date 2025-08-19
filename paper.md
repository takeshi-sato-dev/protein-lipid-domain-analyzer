---
title: 'Protein-Lipid Domain Analyzer: A configurable toolkit for studying membrane protein-lipid domain interactions using proximity-based analysis and Bayesian statistics'
tags:
  - Python
  - molecular dynamics
  - membrane biology
  - lipid transport
  - state classification
  - Bayesian statistics
  - cholesterol-rich domains
  - gangliosides
authors:
  - name: Takeshi Sato
    orcid: 0009-0006-9156-8655
    affiliation: 1
affiliations:
 - name: Kyoto Pharmaceutical University
   index: 1
date: 15 August 2025
bibliography: paper.bib
---

# Summary

Protein-Lipid Domain Analyzer is a comprehensive Python toolkit for analyzing how specific membrane lipids (e.g., GM3 gangliosides, cholesterol, sphingomyelin) affect protein localization between cholesterol-rich and disordered membrane domains in molecular dynamics (MD) simulations. The software employs proximity-based analysis to automatically identify protein states from trajectory data, distinguishing between lipid-bound and unbound states across different membrane domains. It then applies hierarchical Bayesian statistics to rigorously quantify transport effects with uncertainty estimates, enabling researchers to determine whether observed effects are statistically significant or merely random fluctuations.

# Statement of need

Membrane lipid composition plays a crucial role in regulating protein function and cellular signaling. Understanding how specific lipids facilitate protein transport between different membrane domains is essential for advancing our knowledge of membrane biology and developing therapeutic strategies. While many computational tools exist for analyzing MD trajectories, there is a lack of specialized software for quantifying lipid-mediated protein transport effects with proper statistical rigor.

Current approaches often rely on simple distance-based metrics or manual analysis, which fail to capture the complex dynamics of membrane domain organization and protein-lipid interactions. This software addresses these limitations by providing:

1. **Automated detection** of all lipid types and proteins from MD trajectory files
2. **Configurable target lipid analysis** supporting any membrane component
3. **Four-state proximity-based classification** combining lipid binding and domain localization
4. **Hierarchical Bayesian statistics** for quantifying transport effects with uncertainty
5. **Publication-ready visualizations** with statistical annotations

The toolkit is particularly valuable for researchers studying ganglioside-mediated transport, cholesterol effects on membrane organization, and lipid raft dynamics.

# Features

## Core Functionality

- **Automatic system detection**: Dynamically identifies all lipid types and proteins from PSF/XTC trajectory files
- **Configurable target lipid**: Supports analysis of any membrane lipid (GM3, cholesterol, sphingomyelin, etc.)
- **Four-state classification**: Non_TARGET_D, Non_TARGET_CS, TARGET_D, TARGET_CS states
- **State classification**: Proximity-based protein state determination and transition probability matrices
- **Bayesian statistics**: Hierarchical modeling of group-level effects with uncertainty quantification
- **Parallel processing**: Multi-core support for large trajectory analysis

## Key Capabilities

### Automated Membrane Domain Detection

The software automatically identifies cholesterol-rich and disordered membrane domains by integrating multiple biophysical parameters including lipid densities and order parameters. This multi-parameter approach provides more robust domain classification than single-metric methods, capturing the complex nature of membrane organization.

### Four-State Classification Framework

A unique feature of this toolkit is its four-state classification system that simultaneously tracks:
- Target lipid binding status (bound/unbound)
- Membrane domain localization (CS-rich/disordered)

This dual tracking enables quantification of how lipid binding affects protein movement between membrane domains, revealing transport mechanisms that would be missed by analyzing these factors separately.

### Statistical Rigor Through Bayesian Analysis

Unlike simple averaging approaches, the software employs hierarchical Bayesian modeling to:
- Quantify transport effects at both individual protein and population levels
- Provide uncertainty estimates through credible intervals
- Determine statistical significance of observed effects

This statistical framework is essential for distinguishing genuine biological effects from random fluctuations in MD trajectories.

### Flexible Configuration System

The toolkit features automatic detection of all system components from trajectory files, with configurable parameters for:
- Target lipid selection (GM3, cholesterol, sphingomyelin, or any custom lipid)
- Distance thresholds for molecular interactions
- Domain classification criteria
- Analysis frame ranges and sampling rates

This flexibility allows researchers to adapt the analysis to diverse membrane systems and research questions.

## Output and Visualization

The analysis generates three integrated publication-ready figures that together tell the complete story of lipid-mediated transport:

![Combined analysis output showing protein state distributions, lipid transport effects, and group-level Bayesian analysis. (A) State distribution plot reveals the time each protein spends in different membrane states, with percentage labels showing exact proportions. (B) Transport effect summary quantifies how target lipid binding enhances movement to cholesterol-rich domains, with positive values indicating facilitated transport. (C) Group-level Bayesian analysis shows the population effect with credible intervals, demonstrating statistical significance across multiple proteins.](figures/combined_output.png)

**Figure 1**: Example output from Protein-Lipid Domain Analyzer showing the three complementary visualizations generated by the software. **(A) State Distribution Plot**: Displays the time fraction each protein spends in the four states, enabling quick identification of dominant protein behaviors and lipid interaction patterns. **(B) Transport Effect Summary**: Quantifies the differential transport probability, revealing whether target lipid binding enhances protein movement to cholesterol-rich domains. **(C) Group-Level Bayesian Analysis**: Provides population-level statistics with uncertainty quantification, essential for determining the biological significance of observed effects across multiple proteins.

# Implementation

The software is implemented in Python 3.8+ and utilizes several key libraries:

- **MDAnalysis** [@michaud-agrawal2011] for trajectory processing
- **PyMC** [@salvatier2016] for Bayesian statistical modeling  
- **NumPy/SciPy** [@harris2020] for numerical computations
- **Matplotlib/Seaborn** for visualization

The modular architecture separates concerns:

- `config.py`: Configuration management with automatic system detection
- `data_loader.py`: MD trajectory loading and component identification
- `lipid_analysis.py`: Lipid order parameters and domain mapping
- `hmm_analysis.py`: State classification and transition analysis
- `plotting.py`: Publication figure generation
- `analysis_pipeline.py`: Complete workflow coordination

# Usage Example

Test data and example trajectories are available on Zenodo (DOI: 10.5281/zenodo.16888741) for users to explore the software capabilities with realistic molecular dynamics data.

```bash
# Basic usage with automatic detection
python main.py

# Specify target lipid
python main.py --target-lipid CHOL --output cholesterol_analysis

# Custom trajectory files
python main.py --topology system.psf --trajectory traj.xtc --target-lipid GM3

# Use configuration file
python main.py --config custom_config.json
```

# Performance and Validation

The software has been tested on molecular dynamics trajectories ranging from 100 ns to 10 μs, processing up to 100,000 frames efficiently through parallel computing support. The modular architecture ensures scalability, with processing time scaling linearly with trajectory length when using multiple CPU cores. Memory usage remains manageable even for large systems with >500,000 atoms through batch processing of trajectory frames.

# Impact and Applications

This toolkit enables researchers to:

- **Quantify lipid-mediated transport** with statistical rigor
- **Compare effects** across different lipid types and protein systems  
- **Generate publication-ready results** with minimal manual intervention
- **Apply consistent methodology** across membrane biology studies

The software has immediate applications in studying ganglioside function, cholesterol effects on membrane proteins, lipid raft dynamics, and drug-membrane interactions.

# Acknowledgements

We acknowledge the MDAnalysis and PyMC development teams for providing the foundational libraries that made this work possible.
We acknowledge contributions and support from Kyoto Pharmaceutical University Fund for the Promotion of Collaborative Research. This work was partially supported by JSPS KAKENHI Grant Number 21K06038.

# References