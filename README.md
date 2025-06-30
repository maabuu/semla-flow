# 3D Molecular Generation Benchmark

A benchmarking framework for evaluating 3D molecular generation methods in computational chemistry and drug design.

## Overview

Unconditional molecular generation is a stepping stone for conditional molecular generation, which is important in de novo drug design. Recent unconditional 3D molecular generation methods report saturated benchmarks, suggesting it is time to re-evaluate our benchmarks and compare the latest models.

This project assesses five recent high-performing 3D molecular generation methods:
- **EQGAT-diff**
- **FlowMol**
- **GCDM**
- **GeoLDM**
- **SemlaFlow**

The evaluation covers both standard benchmarks and chemical and physical validity metrics. Our findings show that the best method, SemlaFlow, achieves a success rate of 87% in generating valid, unique, and novel molecules without post-processing and 92.4% with post-processing.

**Workshop paper**: [3D Molecular Generation Benchmark](https://arxiv.org/pdf/2505.00518)

## Features

- **Comprehensive Evaluation**: Assess molecular generation methods across multiple validity metrics
- **Chemical Analysis**: Evaluate generated molecules for chemical and physical properties
- **Visualizations**: Generate UMAP plots and molecular visualizations
- **Post-processing Pipeline**: Clean and filter generated molecules
- **Batch Processing**: Support for large-scale molecular generation evaluation
- **Interactive Notebooks**: Jupyter notebooks for exploratory analysis

## Installation

### Prerequisites

- Python >= 3.11
- Conda or pip for package management

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd 3d_mol_gen_benchmark
```

2. Install dependencies using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

## Usage

### Basic Evaluation

Run molecular evaluation on generated molecules:

```bash
python evaluate_molecules.py --input molecules.sdf --output results.csv
```

### Post-processing

Clean and filter generated molecules:

```bash
python postprocess_molecules.py --input raw_molecules.sdf --output clean_molecules.sdf
```

### Batch Processing

Use the provided shell scripts for complete evaluation pipelines:

```bash
# Combine predictions from multiple methods
bash a_combine_predictions.sh

# Run post-processing
bash b_run_postprocessing.sh

# Run comprehensive evaluation
bash c_run_evaluation.sh

# Calculate FCD scores
bash d_run_fcd.sh
```

### Interactive Analysis

Explore the data using Jupyter notebooks:

```bash
jupyter lab
```

Key notebooks:
- `01_analysis.ipynb` - Main analysis and results
- `02_umap_chem_space.ipynb` - Chemical space visualization
- `03_mol_pics.ipynb` - Molecular structure visualization
- `05_fcd_table.ipynb` - FCD score calculations

## Evaluation Metrics

The framework evaluates generated molecules using:

- **Validity**: Chemical validity of generated structures
- **Uniqueness**: Diversity of generated molecules
- **Novelty**: Comparison against known molecular databases
- **Drug-likeness**: QED and other pharmaceutical metrics
- **Physical Properties**: LogP, molecular weight, etc.
- **3D Geometry**: Bond lengths, angles, and conformational quality
- **FCD Scores**: Fréchet ChemNet Distance for distribution comparison

## Project Structure

```
├── data/                    # Dataset storage
├── plots/                   # Generated visualizations
├── tables/                  # Results and evaluation tables
├── archive/                 # Archived analysis notebooks
├── evaluate_molecules.py    # Main evaluation script
├── postprocess_molecules.py # Molecule cleaning utilities
├── tools.py                # Utility functions
├── fcp.py                  # FCD calculation utilities
└── *.ipynb                 # Analysis notebooks
```

## Results

The benchmark reveals significant differences in molecular generation quality across methods:

- **SemlaFlow**: Best overall performance (87% success rate, 92.4% with post-processing)
- **EQGAT-diff, FlowMol, GCDM, GeoLDM**: Varying performance across different metrics

Detailed results and analysis are available in the project notebooks and generated tables.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Dependencies

Key dependencies include:
- RDKit for molecular manipulation
- scikit-learn for machine learning utilities
- pandas/numpy for data processing
- plotly/seaborn for visualization
- FCD for molecular distribution comparison

See `pyproject.toml` for the complete dependency list.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{
buttenschoen2025an,
title={An evaluation of unconditional 3D molecular generation methods},
author={Martin Buttenschoen and Yael Ziv and Garrett M Morris and Charlotte Deane},
booktitle={ICLR 2025 Workshop on Generative and Experimental Perspectives for Biomolecular Design},
year={2025},
url={https://arxiv.org/pdf/2505.00518}
}
```

## Contact

- **Issues & Questions**: Please [open an issue](../../issues) on GitHub
- **Contributions**: Submit a [pull request](../../pulls) or open an issue to discuss
- **Other inquiries**: Contact via email (see paper for contact details)
