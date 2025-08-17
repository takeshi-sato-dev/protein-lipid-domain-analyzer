#!/bin/bash

# Script to prepare test data for Zenodo upload

echo "Preparing test data for Zenodo upload..."

# Create directory for Zenodo upload
ZENODO_DIR="zenodo_test_data"
mkdir -p $ZENODO_DIR

# Copy test files (if they exist)
if [ -f "test_system.psf" ]; then
    cp test_system.psf $ZENODO_DIR/
    echo "Copied test_system.psf"
fi

if [ -f "test_trajectory.xtc" ]; then
    cp test_trajectory.xtc $ZENODO_DIR/
    echo "Copied test_trajectory.xtc"
fi

# Copy example configuration
cp example_config.json $ZENODO_DIR/

# Create README for test data
cat > $ZENODO_DIR/README.md << EOF
# Lipid Domain Transport Analyzer - Test Data

This dataset contains example molecular dynamics trajectories for testing the Lipid Domain Transport Analyzer software.

## Contents

- \`test_system.psf\` - Topology file containing protein and membrane system
- \`test_trajectory.xtc\` - Compressed trajectory file (40 frames)
- \`example_config.json\` - Example configuration file

## Usage

1. Download all files to your working directory
2. Run the analysis:
   \`\`\`bash
   python main.py --topology test_system.psf --trajectory test_trajectory.xtc
   \`\`\`

## System Description

The test system contains:
- 4 transmembrane proteins
- Mixed lipid bilayer including:
  - DIPC (phosphatidylcholine)
  - DOPS (phosphatidylserine)
  - DPSM (sphingomyelin)
  - CHOL (cholesterol)
  - DPG3 (GM3 ganglioside)
- Water and ions

## Software Repository

GitHub: https://github.com/tsato-kyoyaku/lipid-domain-transport-analyzer

## Citation

If you use this test data, please cite both:
- The software paper (JOSS, 2025)
- This Zenodo dataset

## License

This test data is provided under the MIT License.
EOF

# Create citation file
cat > $ZENODO_DIR/CITATION.cff << EOF
cff-version: 1.2.0
title: "Lipid Domain Transport Analyzer - Test Data"
authors:
  - family-names: Sato
    given-names: Takeshi
    orcid: "https://orcid.org/0009-0006-9156-8655"
type: dataset
repository-code: "https://github.com/tsato-kyoyaku/lipid-domain-transport-analyzer"
license: MIT
EOF

# Create zip file for upload
zip -r lipid_domain_transport_test_data.zip $ZENODO_DIR/

echo "Test data prepared in $ZENODO_DIR/"
echo "Zip file created: lipid_domain_transport_test_data.zip"
echo ""
echo "Next steps:"
echo "1. Go to https://zenodo.org/deposit/new"
echo "2. Upload lipid_domain_transport_test_data.zip"
echo "3. Fill in metadata (title, authors, description)"
echo "4. Get DOI and update README.md and DATA_AVAILABILITY.md"