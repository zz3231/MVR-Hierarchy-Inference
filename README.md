# MVR Hierarchy Inference Tool

Interactive Streamlit application for testing MVR (Minimum Violation Ranking) algorithm robustness against selection bias in career transition data.

## Overview

This tool implements the methodology from Huitfeldt et al. (2023) to identify organizational hierarchies from worker flow data. It addresses concerns about whether data limitations affect the algorithm's ability to correctly identify hierarchy levels.

## Features

### Page 1: Synthetic Company Builder
- Configure firm structure (departments and ranks)
- Generate ground truth worker trajectories
- Apply selection bias to simulate LinkedIn-style data limitations

### Page 2: MVR Analysis
- Multiple algorithm variants (paper-exact and alternatives)
- Configurable graph construction, ranking, and clustering methods
- Compare results on ground truth vs biased data
- Detailed visualizations and metrics

### Page 3: Sensitivity Analysis
- Leave-One-Job-Out (LOJO) testing
- Identify critical jobs for hierarchy identification
- Compare algorithm robustness across configurations
- Export results to Excel

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run mvr_app_multipage.py
```

Access at http://localhost:8501

## Algorithm Variants

The tool supports paper-exact implementations and alternatives for robustness testing:

**Directed Graph**: Consecutive pairs (paper) vs All pairs (alternative)

**Initial Ranking**: Unweighted out-degree (paper) vs Weighted (alternative)

**K-means Threshold**: Bonhomme exact (paper) vs Scaled variants

**ILM Pruning**: Optional Algorithm 1 pruning with configurable threshold

## Documentation

- `ALGORITHM_DOCUMENTATION.md` - Technical specifications and paper references
- `PAGE3_SENSITIVITY_ANALYSIS.md` - Sensitivity analysis guide
- `WORKFLOW.md` - Testing protocols and decision points

## Research Question

Can MVR correctly identify organizational hierarchy levels when:
- Lower ranks have 50% observation rates
- Higher ranks have 100% observation rates
- Entire layers or specific jobs are missing

## References

Huitfeldt, I., Kostøl, A. R., Nimczik, J., & Weber, A. (2023). Internal labor markets: A worker flow approach. IZA Discussion Paper No. 14637.

## Contact

For questions or issues, please open a GitHub issue.
