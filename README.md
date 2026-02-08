# DANCE: Dual Adaptive Noise-Cancellation and Enhancement for One-Dimensional Signals


## 🚀 Quick Start

Follow these steps to set up the environment and run the model.

### 1. Installation

**Prerequisite:** Ensure your Python version is **3.10**.

We recommend creating a virtual environment (e.g., using Conda) to manage dependencies:

```bash
conda create -n dance python=3.10
conda activate dance
```

Then, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Please download the following datasets from PhysioNet:
- MIT-BIH Arrhythmia Database:
https://physionet.org/content/mitdb/
- MIT-BIH Noise Stress Test Database:
https://physionet.org/content/nstdb/

### 📂 Directory Structure
After downloading and unzipping, place the dataset folders in the top-level directory (DANCE/). Your project structure should look like the tree below:

```txt
DANCE/
├── mit-bih-arrhythmia-database/         <-- Place Dataset 1 here
├── mit-bih-noise-stress-test-database/  <-- Place Dataset 2 here
├── datasets/
│   ├── split_manager.py
│   └── ...
├── script/
│   └── DANCE.sh
├── requirements.txt
└── README.md
```

**Note**: Ensure the folder names match the structure above so the scripts can locate the files correctly.

### 3. Preprocessing
Run the data manager script to preprocess and split the data:

```bash
python datasets/split_manager.py
```

### 4. Training & Usage
To run the model, execute the provided shell script:

```bash
bash script/DANCE.sh
```

*Alternatively, you can modify script/DANCE.sh to customize parameters for your specific instances.*

---
> **Dan Liu, Tianhai Xie @ IIP-2025**