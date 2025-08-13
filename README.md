# Pragmatics Research Project

Sandy Thomas Pragmatics Research

## Project Overview
This project contains machine learning models and datasets for pragmatics research, including sarcasm detection and semantic analysis.

## Setup

### 1. Virtual Environment
The project uses a Python virtual environment located in `.venv/`. To activate it:

```bash
source .venv/bin/activate
```

### 2. Dependencies
Install required packages from requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Project Structure
```
Pragmatics/
├── .venv/                    # Virtual environment
├── datasets/                 # Training and test datasets
│   ├── train-balanced-sarcasm.csv
│   ├── test-balanced.csv
│   ├── test-unbalanced.csv
│   └── text8
├── model.py                  # Main model implementation
├── semantics_replication.py  # Semantics research script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Usage

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Run the main model:
   ```bash
   python model.py
   ```

3. Run semantics replication:
   ```bash
   python semantics_replication.py
   ```

## Development

- Python 3.12.9
- All dependencies are managed via pip and requirements.txt
- Use the virtual environment for all development work 
