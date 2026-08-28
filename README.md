# Particle Track Momentum Reconstructor

[![Tests](https://github.com/Wafa-Mamaani/Particle-Momentum-Reconstructor/actions/workflows/tests.yml/badge.svg)](https://github.com/Wafa-Mamaani/Particle-Momentum-Reconstructor/actions/workflows/tests.yml)

This repository contains the software project developed for the Software and Computing exam of the M.Sc. in Applied Physics at the University of Bologna.

## Key Features

- Modular pipeline for simulation, preprocessing, training, prediction, and evaluation
- Reproducible model training through controlled random seeds
- Input validation and informative error handling across the main workflow
- Unit and integration testing, including a complete end-to-end pipeline test
- Automated testing with GitHub Actions
- Support for dynamically inferred model input dimensions
- Leakage-free preprocessing using statistics derived only from the training set

## Contents

- [Key Features](#key-features)
- [Survey](#survey)
- [Repository Contents](#repository-contents)
- [Installation](#installation)
- [Tutorial](#tutorial)
    - [Command-Line Options](#command-line-options)
    - [Data Preparation & Preprocessing](#data-preparation--preprocessing)
    - [Training](#training)
    - [Prediction](#prediction)
    - [Evaluation](#evaluation)
- [Testing](#testing)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [References](#references)

## Survey
Charged-particle tracking in magnetic fields is an important component of many High Energy Physics (HEP) experiments. When a charged particle moves through a magnetic field, the Lorentz force bends its trajectory in the plane transverse to the field. The radius of curvature of this trajectory is related to the particle's transverse momentum, making precise tracking measurements essential for momentum reconstruction.

Low-mass tracking detectors, such as straw-tube trackers, record a sequence of spatial hit positions along a particle trajectory while minimizing interactions with detector material. The Mu2e experiment at Fermilab is one example: its straw-tube tracker operates inside the Detector Solenoid and provides the primary momentum measurement for conversion-electron candidates. <sup>[<a href="#ref-1">1</a>]</sup>

This project implements a simplified machine-learning pipeline for transverse-momentum reconstruction from simulated tracking data. Rather than reproducing the full geometry and detector response of a real experiment, the simulator models charged particles moving in a uniform axial magnetic field and records their transverse `(x, y)` intersections with concentric detector layers. Gaussian spatial smearing and stochastic hit inefficiency are included to introduce measurement uncertainty and missing detector hits.

The simulated hit coordinates are preprocessed and used as inputs to a PyTorch multilayer perceptron that predicts the true transverse momentum, `pT`. Performance is evaluated on a held-out test set by comparing the predicted and simulated truth values using residuals, root mean square error (RMSE), bias, and the standard deviation of the residual distribution.

**Example of a straw-tube tracker:** Mu2e detector solenoid (DS) containing the stopping target, tracker and calorimeter, with charged particles passing through a non-zero magnetic field <sup>[<a href="#ref-1">1</a>]</sup>.

<a href="https://github.com/user-attachments/assets/e5b85da3-a522-44b2-bbea-65e23e095cec">
  <img src="https://github.com/user-attachments/assets/e5b85da3-a522-44b2-bbea-65e23e095cec" width="75%">
</a>
  
## Repository Contents
The project is organized into separate modules for each stage of the reconstruction workflow:
```text
├── data_files/                 # Raw simulated CSVs and processed NumPy arrays
    ├── simulated_data
    ├── processed_data
├── results/                    # Reconstruction plots and prediction CSVs
    ├── test_predictions.csv
    ├── plots/
├── tests/                      # Unit and end-to-end pytest suite
│   ├── test_end_to_end.py
│   ├── test_evaluation.py
│   ├── test_model.py
│   ├── test_prediction.py
│   ├── test_preprocessing.py
│   ├── test_simulation.py
│   └── test_training.py
├── weights/                    # Saved model state dictionaries
├── model.py                    # PyTorch MLP architecture with masking logic
├── simulation.py               # MC-truth physics track generator
├── preprocessing.py            # Leakage-free scaling and data splitting
├── train.py                    # PyTorch training engine with early stopping
├── predict.py                  # Inference and physical unit restoration
├── plot.py                     # Final visualization and residual analysis
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Testing and coverage dependencies
└── README.md                   # Project documentation
```
The `data_files/`, `weights/`, and `results/` directories are generated automatically when the pipeline is run and are not tracked by Git.

## Installation
This project was tested with Python 3.12.2. Using a virtual environment is recommended to keep the project dependencies separate from other Python installations.

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Wafa-Mamaani/Particle-Momentum-Reconstructor.git
cd Particle-Momentum-Reconstructor
```

Install the packages required to run the simulation, preprocessing, training, prediction, and evaluation pipeline:

```bash
python -m pip install -r requirements.txt
```

To run the test suite and coverage checks, install the development dependencies instead:

```bash
python -m pip install -r requirements-dev.txt
```

PyTorch installation may depend on the operating system and whether GPU support is required. For platform-specific installation instructions, consult the official PyTorch installation guide:

https://pytorch.org/get-started/locally/

## Tutorial
The reconstruction pipeline is intended to be executed sequentially:
simulation, preprocessing, training, prediction, and evaluation.

### Command-Line Options

Each stage can be configured from the command line without modifying the source code.

#### Simulation

| Option | Default | Description |
| --- | --- | --- |
| `--samples` | `10000` | Number of simulated particle tracks |
| `--seed` | `13` | Random seed for reproducible simulation |
| `--outdir` | `data_files/simulated_data` | Directory for the generated CSV file |

#### Preprocessing

| Option | Default | Description |
| --- | --- | --- |
| `--input` | `data_files/simulated_data/simulated_tracks.csv` | Input simulated-track CSV |
| `--outdir` | `data_files/processed_data` | Directory for processed NumPy arrays |
| `--seed` | `13` | Random seed used for data splitting |

#### Training

| Option | Default | Description |
| --- | --- | --- |
| `--data` | `data_files/processed_data` | Directory containing the processed arrays |
| `--weights` | `weights` | Directory for saved model weights |
| `--epochs` | `150` | Maximum number of training epochs |
| `--batch` | `64` | Training batch size |
| `--lr` | `0.001` | Adam optimizer learning rate |
| `--patience` | `15` | Epochs without validation improvement before early stopping |
| `--seed` | `13` | Random seed for reproducible training |

#### Prediction

| Option | Default | Description |
| --- | --- | --- |
| `--data` | `data_files/processed_data` | Directory containing the processed test data |
| `--weights` | `weights/best_model.pth` | Path to the trained model weights |
| `--outdir` | `results` | Directory for the prediction CSV |

#### Evaluation

| Option | Default | Description |
| --- | --- | --- |
| `--input` | `results/test_predictions.csv` | Prediction CSV to evaluate |
| `--outdir` | `results/plots` | Directory for generated evaluation plots |

### Data Preparation & Preprocessing
To provide a reproducible dataset without depending on experiment-specific simulation software or external data files, this project includes a self-contained toy Monte Carlo track generator.

Generate 50,000 simulated charged-particle tracks across 36 detector layers. The simulator applies Gaussian spatial smearing to model measurement uncertainty and stochastic hit inefficiency to represent missing detector hits.

```bash
python simulation.py --samples 50000 --seed 13
```

Next, preprocess the simulated data for neural-network training. The preprocessing stage splits the dataset into training, validation, and test subsets and calculates scaling parameters using only the training data to avoid information leakage.

```bash
python preprocessing.py --input data_files/simulated_data/simulated_tracks.csv
```

### Training
Train the momentum regressor using PyTorch. The model uses the Adam optimizer and Mean Squared Error (MSE) loss.
```bash
python train.py --epochs 150 --lr 0.001
```

### Prediction
Evaluate the model on the held-out test set. This restores the scaled outputs to physical units (MeV/c).
```bash
python predict.py
```

### Evaluation
Analyze the reconstruction resolution.
```bash
python plot.py
```
The resulting plots in `results/plots/` report the root mean square error (RMSE), bias, and standard deviation of the residual distribution for the reconstructed transverse momentum.


## Testing
The project includes unit and integration tests for the simulation, preprocessing, model, training, prediction, and evaluation stages. An end-to-end test also verifies that the complete reconstruction workflow runs successfully from track simulation through final evaluation.

Install the development dependencies with:

```bash
python -m pip install -r requirements-dev.txt
```

Run the complete test suite with:

```bash
python -m pytest
```

To measure code coverage across the main project modules:

```bash
coverage erase
coverage run --source=simulation,preprocessing,model,train,predict,plot -m pytest
coverage report -m
```
## Assumptions and Limitations

This project is intentionally designed as a simplified and reproducible demonstration of a particle-track momentum reconstruction workflow rather than a complete detector simulation.

The main assumptions and limitations are:

- Particle motion is modeled in the transverse `(x, y)` plane under a uniform axial magnetic field.
- Detector geometry is represented by concentric tracking layers rather than the full geometry of a specific experiment.
- Measurement uncertainty is approximated using Gaussian spatial smearing, while missing hits are introduced through stochastic detector inefficiency.
- The simulation does not model the complete set of physical effects present in a real tracking detector, such as detailed material interactions, multiple scattering, or energy loss.
- The neural network reconstructs transverse momentum (`pT`) from simulated hit coordinates and is not intended as a validated reconstruction model for experimental data.
- Missing detector hits are represented using a fixed padding value and masked before the data are passed through the neural network.
- The prediction stage evaluates the held-out processed test set produced by the preprocessing pipeline; it is not intended as a general-purpose interface for arbitrary raw detector data.

These simplifications keep the project computationally lightweight and allow the complete workflow to remain reproducible and independently testable.

## References
<blockquote id="ref-1">[1] George Gollin Group. Research in high energy physics, primarily concerning the Mu2e experiment at Fermilab. https://hep.physics.illinois.edu/home/g-gollin/research/ </blockquote>
