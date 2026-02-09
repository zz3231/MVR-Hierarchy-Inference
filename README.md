# MVR Hierarchy Inference Tool

Interactive application for testing MVR algorithm robustness against selection bias in career transition data.

## Overview

This tool addresses concerns about whether data limitations (missing observations, biased sampling) affect the Minimum Violation Ranking algorithm's ability to correctly identify organizational hierarchies.

## Features

### Page 1: Synthetic Company Builder
- Configure firm structure (departments, ranks)
- Generate ground truth company with worker trajectories
- Apply observation bias to simulate data limitations
- Visualize firm structure and bias effects

### Page 2: MVR Analysis
- Run MVR algorithm on both ground truth and biased companies
- Compare identified hierarchy levels (K)
- View detailed convergence, variance, and clustering results
- Assess algorithm robustness to selection bias

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run mvr_app_multipage.py
```

Access at `http://localhost:8501`

## Usage Workflow

1. **Configure Firm Structure**
   - Set departments and ranks per department
   - View hierarchy visualization

2. **Generate Ground Truth Company**
   - Set number of workers, promotion rates, time range
   - Generate complete career trajectories

3. **Apply Selection Bias**
   - Configure observation rates by rank level
   - Simulate LinkedIn data limitations
   - Compare ground truth vs observed counts

4. **Run MVR Analysis**
   - Algorithm runs on both companies in parallel
   - Compare identified K values and detailed metrics
   - Assess robustness with convergence plots and statistics

## Research Question

Can MVR correctly identify organizational hierarchy levels when:
- Lower ranks have 50% observation rates
- Higher ranks have 100% observation rates
- Entire layers may be missing
- Promotion reporting is biased by level

## K-means Methods

Three methods for determining optimal hierarchy levels (K):

1. **Average Optimal Ranking Variance** (Default)
   - Threshold: `(N/(N-1)) × Var(average_ranks)`
   - Measures overall dispersion between job positions
   - More aggressive clustering (tends to produce larger K)
   - Formula: `inertia ≤ (1/(N-1)) × Var(avg_ranks) × N`

2. **Bonhomme et al. (2019) - Paper Method**
   - Threshold: `(N/(N-1)) × Σ Var(r_v)` where `Var(r_v)` is variance of job v's rank across optimal rankings
   - Measures within-job position uncertainty
   - More conservative clustering (tends to produce smaller K)
   - Exactly matches paper's Equation B2 (Appendix B.3)
   - Formula: `inertia ≤ (1/(N-1)) × Σ Var(r_v) × N`

3. **Elbow Method (Manual)**
   - User manually selects K after viewing convergence and variance plots
   - Useful for exploratory analysis

**For Academic Publication**: Use Method #2 (Bonhomme) as it matches the paper exactly.

**For Practical Use**: Method #1 provides more fine-grained hierarchies which may be useful for detailed analysis.

## Algorithm Details

### MVR (Minimum Violation Ranking)
- **Paper Method**: Unweighted violation counting (each edge counts as 1 violation)
- Random swap optimization (MCMC algorithm)
- Accepts swaps with S' ≤ S (equal or fewer violations)
- Finds all global optimal rankings when multiple exist
- Samples from the set of minimum violation rankings to create consensus ranking

### ILM Network Pruning (Paper's Algorithm 1)
- **Purpose**: Remove measurement error in occupational coding
- **Method**: Leave-X-percent-out procedure (default X=10%)
- Identifies articulation points (cut vertices) in bipartite worker-job graph
- Removes workers whose removal affects less than X% of jobs
- Extracts largest connected component as the main Internal Labor Market (ILM)
- **Scale-invariant**: Same classification for firms with same structure but different sizes

### Graph Construction
- **Step 1**: Build bipartite graph (workers ↔ jobs)
- **Step 2**: Apply ILM pruning to remove measurement error
- **Step 3**: Extract largest connected component (largest ILM)
- **Step 4**: Build directed graph using ALL PAIRS method from largest ILM
  - Connects all positions in each worker's career path
  - Consecutive duplicates removed (worker staying in same role)
  - Edge weights represent transition frequencies (but not used in violation counting)

### K-means Clustering with Factorization
- After MVR, jobs are grouped into K hierarchy levels
- **Factorization**: Layer labels are renumbered to be consecutive (0, 1, 2, ...)
- Ensures interpretable and consistent layer numbering across different runs

## Reference

Based on: Huitfeldt et al. (2023) - "Internal labor markets: A worker flow approach"
