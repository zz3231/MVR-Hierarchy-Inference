# MVR Hierarchy Inference Tool

Interactive web application for testing MVR algorithm robustness against selection bias in career transition data.

## Overview

This tool addresses reviewer concerns about whether LinkedIn data limitations (missing observations, biased sampling) affect the Minimum Violation Ranking algorithm's ability to correctly identify organizational hierarchies.

## Features

### Page 1: Synthetic Company Builder
- Configure firm structure (departments, ranks)
- Set promotion probabilities
- Generate ground truth company
- Apply observation bias to simulate LinkedIn data issues

### Page 2: MVR Analysis
- Run MVR algorithm on both companies
- Compare identified hierarchy levels (K)
- Test algorithm robustness

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run mvr_app_multipage.py
```

Access at `http://localhost:8501`

## Project Structure

```
corp/
├── mvr_app_multipage.py          # Main multi-page Streamlit app
├── firm_structure.py             # Company generation utilities
├── firm_structure_simulator.ipynb # Development notebook
├── mvr_website.py                # Legacy single-page app
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Usage Workflow

1. **Configure Firm**
   - Set number of departments
   - Define ranks per department
   - View hierarchy visualization

2. **Generate Ground Truth**
   - Set total workers
   - Set promotion rates
   - Generate complete trajectories

3. **Apply Selection Bias**
   - Configure observation rates by rank
   - Simulate LinkedIn data limitations
   - View comparison table

4. **Run MVR Analysis**
   - Algorithm runs on both companies
   - Compare identified K values
   - Assess robustness

## Research Question

Can MVR correctly identify organizational hierarchy levels when:
- Lower ranks have 50% observation rates
- Higher ranks have 100% observation rates
- Entire layers may be missing
- Promotion reporting is biased by level

## Contact

Based on: Huitfeldt et al. (2023) - "Internal labor markets: A worker flow approach"
