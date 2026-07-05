## 更新后的 README.md（使用 environment.yml）

---

# FlashAIR: Fast Infrared Spectral Analysis with Physical Prior and Residual Correction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![RDKit](https://img.shields.io/badge/RDKit-2023.09+-orange.svg)](https://www.rdkit.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-red.svg)](https://www.tensorflow.org/)
[![GPflow](https://img.shields.io/badge/GPflow-2.9+-purple.svg)](https://www.gpflow.org/)

**FlashAIR** (Fast Infrared Spectral Analysis with Physical Prior and Residual Correction) is a comprehensive tool for predicting infrared (IR) spectra from molecular structures. It provides two complementary approaches:

- **FlashAIRa**: Fast fragment-based IR spectrum prediction using molecular fragmentation and library matching
- **FlashAIRx**: High-accuracy IR spectrum correction using GFN2-xTB calculations with Gaussian Process Regression (GPR) residual correction

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [FlashAIRa: Fast Fragment-based Prediction](#flashaira-fast-fragment-based-prediction)
  - [FlashAIRx: Physics-based Correction](#flashairx-physics-based-correction)
- [Model Files](#-model-files)
- [Demo](#-demo)
- [Citation](#-citation)
- [License](#-license)

---

## ✨ Features

### FlashAIRa
- **Fast prediction** without quantum chemical calculations
- **Molecular fragmentation** based on functional group expansion
- **Four-level matching** (L1: exact, L2: similarity, L3: functional group, L4: aromatic)
- Suitable for **high-throughput screening** and rapid analysis

### FlashAIRx
- **Physics-based prior** from GFN2-xTB calculations
- **GPR residual correction** trained on high-level DFT data (FlashAIR-QM9d)
- **Two-stage calibration**: XTB → DFT → EXP (optional)
- **High accuracy** with interpretable physical basis

### Common Features
- **SMILES and 3D coordinate input** support
- **Multiple similarity metrics**: PCC, Spearman, SIS
- **Mixture analysis**: Identify molecular mixtures from experimental spectra
- **Jupyter Notebook demo** for easy testing and visualization

---

## 📦 Installation

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python 3.9 or higher

### Step 1: Create Conda Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/FlashAIR.git
cd FlashAIR

# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate flashair

# Register Jupyter kernel (optional)
python -m ipykernel install --user --name=flashair --display-name="Python (flashair)"
```

### Step 2: Install xTB (Optional, for FlashAIRx)

```bash
# Option 1: Using conda
conda install xtb -c conda-forge

# Option 2: Download from https://github.com/grimme-lab/xtb/releases
```

### Step 3: Verify Installation

```bash
python -c "
import rdkit, numpy, tensorflow, gpflow
print('✅ All dependencies installed successfully!')
print(f'RDKit: {rdkit.__version__}')
print(f'NumPy: {numpy.__version__}')
print(f'TensorFlow: {tensorflow.__version__}')
print(f'GPflow: {gpflow.__version__}')
"
```

---

## 📁 Project Structure

```
FlashAIR/
├── flashair/                          # Main package
│   ├── __init__.py                    # Package initialization
│   ├── config.py                      # Configuration settings
│   ├── FlashAIRa_Predictor.py         # FlashAIRa: Fast fragment-based prediction
│   ├── FlashAIRa_analysis.py          # FlashAIRa evaluation tools
│   ├── FlashAIRa_prediction_and_evaluate.py  # FlashAIRa core algorithm
│   ├── FlashAIRx_preprocess.py        # FlashAIRx: XTB calculation & preprocessing
│   ├── FlashAIRx_prediction.py        # FlashAIRx: GPR-based correction
│   ├── FlashAIRx_matcher.py           # FlashAIRx: Spectral matching
│   ├── FlashAIRx_train_and_evaluate.py # FlashAIRx training pipeline
│   └── models/                        # Pre-trained model files
│       ├── fgss_single.pkl            # Single functional group models
│       ├── fgsm_multi.pkl             # Multiple functional group models
│       ├── xtb2dft_all_models.pkl     # Complete GPR model dictionary
│       ├── Acid_gpr.pkl               # Carboxylic acid specific model
│       ├── Halide_gpr.pkl             # Halide specific model
│       └── Nitro_gpr.pkl              # Nitro specific model
├── data/                              # Database
│   └── FlashAIR-QM9d.db               # QM9-derived IR spectra database
├── Demo/                              # Jupyter demo notebooks
│   └── FlashAIR_Demo.ipynb            # Main demonstration notebook
├── environment.yml                    # Conda environment specification
└── README.md                          # This file
```

---

## 🚀 Quick Start

### Run the Demo Notebook

```bash
# Activate environment
conda activate flashair

# Navigate to Demo directory
cd Demo

# Launch Jupyter Lab
jupyter lab FlashAIR_Demo.ipynb
```

The demo notebook will guide you through:
1. Environment verification
2. FlashAIRa prediction examples
3. FlashAIRx prediction examples
4. Custom molecule prediction
5. Spectral matching and mixture analysis

---

## 📖 Usage

### FlashAIRa: Fast Fragment-based Prediction

```python
from flashair import FlashAIRaPredictor

# Initialize predictor (test_mode uses base library for faster loading)
predictor = FlashAIRaPredictor(
    db_path="data/FlashAIR-QM9d.db",
    test_mode=True
)

# Predict IR spectrum from SMILES
result = predictor.predict_from_smiles("CCO")  # Ethanol

if result['success']:
    print(f"Match level: {result['match_level']}")
    print(f"Similarity: {result['match_similarity']:.4f}")
    predicted_ir = result['predicted_ir']
    wavenumber = result['wavenumber']

# Batch prediction
smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
results = predictor.predict_from_smiles_batch(smiles_list)
```

### FlashAIRx: Physics-based Correction

```python
from flashair import FlashAIRxPredictor
from flashair import preprocess

# Initialize predictor
predictor = FlashAIRxPredictor(
    model_dir="flashair/models",
    model_dict_path="flashair/models/xtb2dft_all_models.pkl"
)

# Step 1: Preprocess SMILES → XTB spectrum
preprocess_result = preprocess(
    input_data="CCO",
    input_type="smiles",
    output_dir="./temp"
)

if preprocess_result['success']:
    xtb_spectrum = preprocess_result['gpr_input']['X']
    functional_groups = preprocess_result['gpr_input']['functional_groups']
    
    # Step 2: Predict DFT spectrum
    result = predictor.predict_dft_only(
        xtb_spectrum=xtb_spectrum,
        functional_groups=functional_groups
    )
    
    if result['success']:
        dft_spectrum = result['dft_spectrum']
        print(f"Matched model: {result['matched_key']}")
```

### Spectral Matching with FlashAIRx Matcher

```python
from flashair import SpectralMatcher, load_user_spectrum

# Initialize matcher
matcher = SpectralMatcher(max_library_size=10)

# Add predicted spectra to library
matcher.add_molecule("CCO", name="Ethanol")
matcher.add_molecule("CC(=O)O", name="Acetic Acid")

# Load experimental spectrum
user_spectrum = load_user_spectrum("experimental_spectrum.csv")

# Match against library
result = matcher.match(user_spectrum['intensity_norm'])

# Get best match
if result['best_overall']:
    print(f"Best match: {result['best_overall']['name']}")
    print(f"PCC: {result['best_overall']['similarity']['pcc']:.4f}")
```

---

## 📊 Model Files

| File | Description | Size |
|------|-------------|------|
| `xtb2dft_all_models.pkl` | Complete dictionary of all GPR models | ~1.4 GB |
| `fgss_single.pkl` | Single functional group models (16 types) | ~300 MB |
| `fgsm_multi.pkl` | Multiple functional group models | ~1.1 GB |
| `Acid_gpr.pkl` | Carboxylic acid specific model | ~10 MB |
| `Halide_gpr.pkl` | Halide specific model | ~19 MB |
| `Nitro_gpr.pkl` | Nitro specific model | ~1.2 MB |

The models are pre-trained on the FlashAIR-QM9d database and cover 16 functional group types:
- Alkane, Alkene, Alkyne, Alcohol, Aldehyde, Ketone, Carboxylic Acid, Ester, Ether, Amide, Amine, Imine, Nitrile, Nitro, Halide, Aromatic

---

## 📝 Citation

If you use FlashAIR in your research, please cite:

```bibtex
@article{flashair2024,
    title={FlashAIR: Fast and Interpretable IR Spectral Analysis via Physical Prior and Residual Correction},
    author={FlashAIR Team},
    journal={},
    year={2024},
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the FlashAIR team.

---

## 🙏 Acknowledgments

- The QM9 dataset for providing molecular data
- The xtb developers for the GFN2-xTB method
- The GPflow team for the Gaussian Process framework

---

**FlashAIR: Bridging the gap between speed and accuracy in IR spectroscopy** 🔬✨
