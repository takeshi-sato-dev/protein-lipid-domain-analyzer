# Protein-Lipid Domain Analyzer

A comprehensive analysis toolkit for studying protein-lipid domain interactions and target lipid-mediated protein transport to cholesterol-rich membrane domains using proximity-based analysis and Bayesian statistics on molecular dynamics simulation data.

## Overview

This package analyzes how target lipids (e.g., GM3 gangliosides, DPG3, or other membrane components) affect protein localization between cholesterol/sphingomyelin-rich (CS-rich) and disordered (D-rich) membrane domains. It uses:

- **Proximity-based analysis** to identify protein states based on target lipid binding and domain localization
- **Bayesian statistics** to quantify target lipid effects on protein transport
- **Configurable target lipids** for flexible analysis of different membrane components

## Features

- ✅ **Configurable target lipid analysis** (GM3, DPG3, DPGS, etc.)
- ✅ **Four-state classification**: Non_TARGET_D, Non_TARGET_CS, TARGET_D, TARGET_CS
- ✅ **Proximity-based state classification** and transition analysis
- ✅ **Hierarchical Bayesian modeling** of group-level effects
- ✅ **Publication-ready figures** (PNG + SVG output)
- ✅ **Parallel processing** support for large trajectories
- ✅ **Modular architecture** for easy extension and customization

## Installation

### Prerequisites

- Python 3.10+
- MDAnalysis
- PyMC
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- ArviZ (for Bayesian analysis)

### Install dependencies

```bash
pip install MDAnalysis pymc numpy scipy pandas matplotlib seaborn arviz
```

## Quick Start

### Basic Usage

```bash
# Run with default settings (DPG3 as target lipid)
python main.py

# Specify target lipid
python main.py --target-lipid GM3

# Custom output directory
python main.py --output my_analysis_results --target-lipid DPGS

# Use configuration file
python main.py --config my_config.json
```

### Required Files

Test data and example trajectories are available on Zenodo due to file size:
- **Zenodo DOI**: 10.5281/zenodo.16900023
- **Download**: See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)

Required files:
- Topology file (PSF/PDB/GRO format)
- Trajectory file (XTC/DCD/TRR format)

### Configuration

The software **automatically detects** all lipid types and proteins from your trajectory files. You can optionally create a configuration file to customize analysis parameters:

```python
# Generate example configuration
from src.config import create_example_config
create_example_config("my_config.json")
```

Example config.json:
```json
{
  "TARGET_LIPID": "GM3",
  "TARGET_LIPID_ALIASES": ["GM3", "gm3", "DPG3", "dpg3"],
  "EXCLUDE_FROM_LIPIDS": ["TIP3", "POT", "CLA", "SOD", "WAT", "Na+", "Cl-"],
  "START_FRAME": 20000,
  "STOP_FRAME": 80000,
  "FRAME_STEP": 10,
  "TARGET_LIPID_INTERACTION_RADIUS": 10.0,
  "CHOL_DENSITY_THRESHOLD": 0.8,
  "SM_DENSITY_THRESHOLD": 0.55
}
```

**Note**: `LIPID_TYPES` and `PROTEIN_NAMES` are automatically detected from your PSF/trajectory files and don't need to be specified in the config.

## Output

The analysis generates three publication figures:

1. **`state_distribution.png/svg`** - Protein state distributions across the four states
2. **`{target_lipid}_effect_summary.png/svg`** - Target lipid transport and conditional effects
3. **`group_effect_size.png/svg`** - Group-level Bayesian effect analysis (multiple proteins)

## Module Structure

```
src/
├── __init__.py              # Package initialization
├── config.py               # Configuration management
├── data_loader.py          # MD trajectory loading and protein identification  
├── lipid_analysis.py       # Lipid order parameters and domain mapping
├── hmm_analysis.py         # Hidden Markov Model state determination
├── plotting.py             # Publication figure generation
└── analysis_pipeline.py    # Main analysis coordinator
```

## Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --output DIR            Output directory for results
  --n-jobs N             Number of parallel jobs (-1 for all CPUs)
  --force                Force overwrite of existing output directory
  --target-lipid LIPID   Target lipid for analysis (e.g., GM3, DPG3, DPGS)
  --config CONFIG        Path to configuration file (JSON format)
```

## Scientific Background

### Four-State Model

The analysis classifies each protein frame into one of four states:

- **Non_TARGET_D**: No target lipid binding, disordered domain
- **Non_TARGET_CS**: No target lipid binding, cholesterol-rich domain  
- **TARGET_D**: Target lipid binding, disordered domain
- **TARGET_CS**: Target lipid binding, cholesterol-rich domain

### Transport Effect Quantification

Transport effects are quantified by comparing transition probabilities:
- **Direct transport**: P(TARGET_D → TARGET_CS) - P(Non_TARGET_D → Non_TARGET_CS)
- **Long-term equilibrium**: Steady-state probability distributions

### Bayesian Analysis

Hierarchical Bayesian modeling estimates:
- Individual protein transport effects
- Group-level (population) effects
- Uncertainty quantification

## Customization

### Adding New Target Lipids

1. Update `TARGET_LIPID_ALIASES` in config.py
2. Ensure lipid is present in trajectory
3. Run analysis with `--target-lipid YOUR_LIPID`

### Modifying Analysis Parameters

Edit configuration file or modify `src/config.py`:
- Frame ranges and sampling
- Interaction radii
- Domain classification thresholds
- HMM parameters

## Troubleshooting

### Common Issues

1. **"Trajectory files not found"**
   - Ensure `step5_assembly.psf` and `md_wrapped.xtc` are in working directory

2. **"No target lipid found"**
   - Check target lipid name matches trajectory residue names
   - Add aliases to configuration

3. **"Insufficient data for Bayesian analysis"**
   - Need ≥2 proteins for group-level analysis
   - Single protein analysis still generates other figures

### Performance Tips

- Use `--n-jobs -1` for parallel processing
- Adjust frame ranges in config for faster testing
- Reduce `FRAME_STEP` for higher temporal resolution

## Citation

If you use this software in your research, please cite:

```bibtex
@article{sato2025protein,
  title={Protein-Lipid Domain Analyzer: A configurable toolkit for studying membrane protein-lipid domain interactions using proximity-based analysis and Bayesian statistics},
  author={Sato, Takeshi},
  journal={Journal of Open Source Software},
  year={2025},
  publisher={The Open Journal},
  note={Submitted for publication}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Support

- 📖 **Documentation**: See [README.md](README.md) for usage instructions
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/takeshi-sato-dev/protein-lipid-domain-analyzer/issues)
- 💬 **Questions**: Open a [GitHub Discussion](https://github.com/takeshi-sato-dev/protein-lipid-domain-analyzer/discussions)

## Acknowledgments

- MDAnalysis development team for trajectory analysis tools
- PyMC development team for Bayesian modeling framework
- The membrane biology research community for inspiration and feedback