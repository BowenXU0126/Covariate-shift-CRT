# Covariate Shift Corrected Conditional Independence Testing (NeurIPS 2024)

This repository contains the codebase for our paper on **Covariate Shift Conditional Randomization Test (CRT)**. The `csPCR` function provides the core implementation of our proposed statistical method.

## 📂 Repository Structure

- **`csPCR_functions.py`** - Contains essential functions for the csPCR statistical method.
- **`Simulation_Demo.ipynb`** - Demonstrates how to use the package on a simulated dataset.
- **`experimental/`** - Contains scripts for running simulations on a server.
- **`Benchmark_functions.py`** - Implementation of our comparison benchmark.
- **`env.txt`** - Environment setup file for dependencies.

## 🔧 Installation & Setup
To use the package, clone the repository and install dependencies:

```bash
git clone <repo-link>
cd <repo-name>
pip install -r env.txt
```

## 🚀 Usage
### Running Simulations
To run a simulation using `csPCR`:
```python
from csPCR_functions import *
# Example usage with a simulated dataset
...
```

For detailed usage, check `Simulation_Demo.ipynb`.

## 📖 Citation
If you use this code in your research, please cite our paper:
[Paper Link](<https://arxiv.org/abs/2405.19231>)

---
For any questions, feel free to open an issue or contact us at <bowenxu@g.harvard.edu>

