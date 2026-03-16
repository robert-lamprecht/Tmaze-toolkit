# TMaze Analysis

A set of tools for running the novel Tmaze Textural Discrimination Task

## Overview

This project provides tools for:
- Extracting motion traces from doors in experimental videos
- Processing and filtering the motion data
- Detecting trial start/end events
- Tracking Mouse Position Across Trials
- Visualizing and analyzing the results

## Repository Structure

```
Tmaze-toolkit/
├── notebooks/      # Jupyter notebooks for analysis
├── src/            # Source code package
├── scripts/        # Runnable scripts
├── tests/          # Unit tests
└── data/           # Data storage (not tracked in git)
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/robert-lamprecht/Tmaze-toolkit.git
cd Tmaze-toolkit.git
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

## Usage

### Extracting Door Traces

Run the extraction script:
```bash
python scripts/extract_door_traces.py
```

This will:
1. Open file dialogs to select video files
2. Guide you through selecting regions of interest
3. Extract motion traces from doors
4. Save the results as pickle files

### Analyzing Results

Open and run the analysis notebook:
```bash
jupyter notebook notebooks/analyze_door_traces.ipynb
```

## Data Format

Door traces are saved as pickled dictionaries with the following structure:
```python
{
    'door1': [...],  # List of motion values for door 1
    'door2': [...],  # List of motion values for door 2
    'door3': [...],  # List of motion values for door 3
    'door4': [...],  # List of motion values for door 4
}
```
