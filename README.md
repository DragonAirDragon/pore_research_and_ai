# Pore Analysis Project

Automated analysis and detection of pores in porous materials using computer vision and deep learning.

## 📁 Project Structure

```
pore-generate/
├── src/                         # Core library code
│   ├── config_loader.py        # Configuration file handler
│   ├── image_processor.py      # Image processing and noise/texture generation
│   └── pore_generator.py       # Procedural pore generation
│
├── scripts/                     # User-facing scripts
│   ├── generate_images.py      # Generate synthetic pore images
│   ├── generate_dataset.py     # Create large training datasets
│   ├── analyze_mask.py         # Analyze pore masks (Watershed method)
│   └── tune_parameters.py      # Optimize Watershed parameters
│
├── models/                      # Neural network models
│   ├── segmentation/           # Binary segmentation UNet (legacy)
│   └── regression/             # Distance map regression UNet
│       ├── model.py            # UNet architecture
│       ├── train.py            # Training script
│       ├── inference.py        # Inference and visualization
│       └── generate_dataset.py # Generate distance map dataset
│
├── tests/                       # Integration tests
│   └── test_integration.py
│
├── data/                        # Data directories (gitignored)
│   ├── synthetic/              # Generated synthetic images
│   ├── datasets/               # Training datasets
│   └── real/                   # Real sample photos
│
└── outputs/                     # Results (gitignored)
    ├── analyzed/               # Analysis results
    ├── visualizations/         # Inference visualizations
    └── checkpoints/            # Model checkpoints
```

## 🚀 Quick Start

### 1. Generate Synthetic Images
```bash
python scripts/generate_images.py
```

### 2. Analyze a Pore Mask (Watershed Method)
```bash
python scripts/analyze_mask.py --input mask.png --output analysis.csv
```

### 3. Train Neural Network (Regression UNet)
```bash
# Generate training dataset with realistic ceramic textures
python models/regression/generate_dataset.py

# Train the model
cd models/regression && python train.py
```

### 4. Run Inference on Real Photos
```bash
python models/regression/inference.py --input your_photo.jpg --output result.png
```

## 🧠 Methods

### Classical: Watershed Segmentation
- Uses distance transform and local maxima detection
- Fast and interpretable
- Best for clean binary masks
- Command: `scripts/analyze_mask.py`

### Neural Network: Distance Map Regression
- Predicts distance maps using UNet
- Robust to noise and overlapping pores
- Trained on realistic synthetic data
- Command: `models/regression/inference.py`

## 📊 Configuration

Edit `config.json` to customize:
- Image size
- Pore count and size distributions
- Noise parameters
- Ceramic texture appearance

## 🔧 Requirements

```bash
pip install -r requirements.txt
```

## 📝 Citation

This tool was developed for automated pore analysis in ceramic and porous materials.
