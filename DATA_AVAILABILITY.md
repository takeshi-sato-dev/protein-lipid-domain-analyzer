# Data Availability

## Test Data and Example Trajectories

Due to the large size of molecular dynamics trajectory files, test data and example trajectories are hosted on Zenodo rather than GitHub.

### Download Test Data

The test dataset includes:
- `test_system.psf` - Topology file for test system
- `test_trajectory.xtc` - Trajectory file (compressed coordinates)
- Example output figures
- Configuration examples

**Zenodo DOI**: 10.5281/zenodo.16888741

**Download link**: https://zenodo.org/records/16888741

### File Sizes
- test_system.psf: ~5 MB
- test_trajectory.xtc: ~50-100 MB (compressed)

### Usage

1. Download the test data from Zenodo
2. Place the files in your working directory
3. Run the analysis:

```bash
python main.py --topology test_system.psf --trajectory test_trajectory.xtc
```

### Creating Your Own Test Data

If you have your own MD trajectories, ensure they include:
- Protein structures
- Membrane lipids (including your target lipid of interest)
- Proper periodic boundary conditions
- Wrapped coordinates (recommended)

### Data Format Requirements

- **Topology**: PSF, PDB, or GRO format
- **Trajectory**: XTC, DCD, or TRR format
- **Simulation requirements**: 
  - Membrane system with identifiable lipids
  - At least one protein
  - Minimum 1000 frames recommended for statistical significance

## Reproducing Paper Figures

To reproduce the exact figures from our paper:

1. Download the full dataset from Zenodo (DOI: 10.5281/zenodo.16888741)
2. Use the provided configuration file: `paper_config.json`
3. Run: `python main.py --config paper_config.json`

## Contact

For questions about data availability or format requirements, please open an issue on GitHub.