# Particle Track Momentum Reconstructor
This repository contains documentation regarding the Software and Computing exam of the Applied Physics curriculum for M.Sc. degree at UniBo.

- [Survey](#Survey)
- [Repository Contents](#Repository-Contents)
- [Installation](#Installation)
- [Tutorial](#Tutorial)
    - [Data Preparation & Preprocessing](#Data-Preparation-&-Preprocessing)
    - [Training](#Training)
    - [Prediction](#Prediction)
    - [Evaluation](#Evaluation)
- [Testing](#Testing)
- [References](#References)

## Survey
Modern bleeding-edge High Energy Physics (HEP) experiments are usually divided into three main catagories refered to as: intensity, energy and cosmic frontiers. Physicists involved in the intensity and energy frontiers, seek to shed light on rare interactions between fundamental particles at various energy scales, which in turn could lead to more precise measurements of established fundamental values, extensions in the Standard Model (SM), or the discovery of new Beyond the Standard Model (BSM) physics. Many such experiments require ultra-sensetive detection systems that come in the form of ultra-light straw-tube or wire-chamber tracking detectors. These detection systems must produce high-resolution reconstruction of signal tracks such that they are distinguished from the overwhelming background events. The vast majority of such detector systems, which study charge particles, take advantage of solenoid-like superconucting magnets that produce uniform magnetic fields, which in turn would force the charged particles to create helical tracks as they traverse the length of the detector, due to the Lorentz force applied in the transverse plane. Detector electronics later try to reconstruct the 3D hit positions of the passing charged particles, which would then be reconstructed into 3D tracks with near-uniform helical radii. At last, the Larmor radius of the helix is used, in tandem with the charge of the passing particle and the magnitude of the uniform magnetic field, to extract the initial transverse momentum of the aforementioned particle. The longitudinal component of the momentum is later recovered via energy deposition values inside the calorimeter and tracker cells.

This project aims to provide a simple machine learning (ML) pipeline in which simulated MC-truth branch (x,y) hit position data for signal electrons are used to predict initial transverse momenta of the said particles. The predicted values are later compared with MC-truth transverse momentum values via the residuals, with per-event standard deviation values acting as metrics for detection resolution.

**Example of a straw-tube tracker:** Mu2e detector solenoid (DS) containing the stopping target, tracker and calorimeter, with charged particles passing through a non-zero magnetic field <sup>[<a href="#ref-1">1</a>]</sup>.

<a href="https://github.com/user-attachments/assets/e5b85da3-a522-44b2-bbea-65e23e095cec">
  <img src="https://github.com/user-attachments/assets/e5b85da3-a522-44b2-bbea-65e23e095cec" width="75%">
</a>

## Repository Contents
The project is organized into decoupled modules to satisfy the "bus test" of software maintainability:
```text
├── data_files/                 # Raw simulated CSVs and processed tensors
    ├── simulated_data
    ├── processed_data
├── results/                    # Reconstruction plots and prediction CSVs
    ├── test_predictions.csv
    ├── plots/
├── tests/                      # Pytest suite with full code coverage
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_simulation.py
├── weights/                    # Saved model state dictionaries
├── model.py                    # PyTorch MLP architecture with masking logic
├── simulation.py               # MC-truth physics track generator
├── preprocessing.py            # Leakage-free scaling and data splitting
├── train.py                    # PyTorch training engine with Early Stopping
├── predict.py                  # Inference and physical unit restoration
├── plot.py                     # Final visualization and residual analysis
├── PACKAGE_DEPENDENCIES.txt    # Dependency list
└── README.md                   # Project documentation
```

## Installation
For the pipeline to run seamlessly, it is required to have Python 3.8+ installed on your machine.


To begine, the user must clone this repository:
```bash
git clone https://github.com/Wafa-Mamaani/Momentum_Reconstruction
```

Aside from Python, it's recommended to create a conda environment to stage the packages and modules of this project.
This project depends on the Pytorch ML framework, aside from a couple of other well-established scientific packages. 
To install them in your conda env it's convinient to run the following:
```bash
pip install -r PACKAGE_DEPENDENCIES.txt
```

Please be warned that PyTorch is quite OS specific; thus, you may want to recheck proper GPU installation guides and commands at the offcial website: 
https://pytorch.org/get-started/locally/

## Tutorial
The model expects a strict sequential execution order.

### Data Preparation & Preprocessing
Since Fermilab GEANT4 simulations of the Mu2e experiment are under strict security locks, we have created a self-confined simulation that will generate the CE electron helical tracks within the magnetic field of the Tracker.
Generate 50,000 simulated tracks with 36 planes of detection. The simulator applies Gaussian smearing to model spatial resolution and stochastic inefficiency to model detector dead-zones.
```bash
python simulation.py --samples 50000 --seed 13
```
Next, we prepare the data for the neural network. This script isolates the training statistics to ensure zero information leakage to the test set.
```bash
python preprocessing.py --input data/simulated_tracks.csv
```

### Training
rain the momentum regressor using PyTorch. The model uses the Adam optimizer and Mean Squared Error (MSE) loss.
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
The resulting plots in `results/plots/` provide the Root Mean Square Error (RMSE) and a residual distribution, which characterizes the detector's momentum resolution.

| Metric | Typical Performance |
| :--- | :--- |
| **Validation MSE** | ~ 0.0024 |
| **Momentum Resolution (Std Dev)** | < 0.6 MeV/c |

## Testing
This codebase maintains **100% test coverage** across all core logic modules (`simulation.py`, `preprocessing.py`, `model.py`). The suite rigorously tests numerical scaling invariants, dynamic padding mask behaviors, and physical kinematic bounds.

To execute the test suite and view the coverage report:
```bash
coverage run -m pytest
coverage report -m
```

## References
<blockquote id="ref-1">[1] George Gollin Group. Research in high energy physics, primarily concerning the Mu2e experiment at Fermilab. https://hep.physics.illinois.edu/home/g-gollin/research/ </blockquote>
