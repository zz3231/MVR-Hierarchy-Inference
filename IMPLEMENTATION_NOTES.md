# Implementation Notes - Paper-Compliant MVR Algorithm

## Summary of Changes

This implementation strictly follows **Huitfeldt et al. (2023) "Internal labor markets: A worker flow approach"** (dp14637.pdf) to ensure academic rigor and reproducibility.

## Key Improvements

### 1. Unweighted Violation Counting (Paper Method)
**Changed from**: Weighted violations (RA's code used edge weights)
```python
# OLD (RA's method)
violations = sum(H[u][v]['weight'] for u, v in H.edges() if rank[u] > rank[v])

# NEW (Paper's method - Algorithm 2)
violations = sum(1 for u, v in H.edges() if rank[u] > rank[v])
```

**Rationale**: Paper explicitly states violations are counted as edges, not weighted by transition frequencies.

### 2. ILM Network Pruning (Paper's Algorithm 1)
**New Feature**: Implemented leave-X-percent-out procedure to handle measurement error

**Algorithm**:
1. Build bipartite graph G(U, V, E) where U=workers, V=jobs
2. Compute degree d_v for each job v
3. Create G' where each edge to job v is duplicated ⌈100/(d_v × X)⌉ times
4. Find articulation points in G'
5. Remove worker articulation points from original G
6. Extract largest connected component as main ILM

**Parameters**:
- X = 10% (default, paper's choice)
- Configurable in UI (5%, 10%, 15%, 20%)

**Impact**: Removes workers whose career paths represent <X% of transitions, likely measurement errors

### 3. Factorized Layer Numbering
**New Feature**: K-means cluster labels are renumbered to be consecutive (0, 1, 2, ...)

```python
# After K-means clustering
labels_sorted = [labels[jobs.index(j)] for j in jobs]
layer_ids = pd.factorize(labels_sorted)[0]  # Creates 0, 1, 2, ... mapping
```

**Rationale**: Ensures interpretable layer numbers and consistent visualization

### 4. Graph Construction Pipeline
**Complete workflow now follows paper exactly**:

1. **Build Bipartite Graph** (Section 3.1, Algorithm 1)
   - Workers (bipartite=0) connected to Jobs (bipartite=1)
   - Represents all observed transitions

2. **Prune Network** (Section 3.1, Algorithm 1, Appendix B.1)
   - Remove articulation point workers affecting <X% of jobs
   - Addresses occupational coding measurement error

3. **Extract Largest ILM** (Section 3.1)
   - Find connected components
   - Select largest component as main Internal Labor Market

4. **Build Directed Graph H** (Section 3.2, ALL PAIRS method)
   - From largest ILM, construct directed job transition graph
   - ALL PAIRS: connect all (i,j) where i<j in career path

5. **MVR Ranking** (Section 3.2.1, Algorithm 2)
   - Minimize violations using random swap MCMC
   - Accept swaps with S' ≤ S

6. **K-means Clustering** (Section 3.2.2, Appendix B.3)
   - Group jobs into K hierarchy levels
   - Choose K where Q(K) ≤ (1/(N-1)) × Σ Var(r_v)

## Differences from RA's Implementation

| Aspect | RA's Code | Paper Method (Current) |
|--------|-----------|------------------------|
| Violation counting | Weighted by edge weights | Unweighted (count edges) |
| ILM pruning | Articulation point removal (buggy) | Leave-X-percent-out procedure |
| Network construction | Direct to directed graph | Bipartite → Prune → ILM → Directed |
| Layer numbering | Raw K-means labels | Factorized consecutive labels |

## Validation

To verify paper compliance:
1. Violation counting matches Algorithm 2 pseudocode (line 2792-2793 in paper)
2. ILM pruning matches Algorithm 1 description (line 2754-2760 in paper)
3. K-means threshold matches Equation B2 (line 2829-2840 in paper)
4. Graph construction follows Section 3.1-3.2 methodology

## Usage Notes

### When to Enable ILM Pruning
- **Enable** (default): For real-world data with potential measurement error
- **Disable**: For synthetic data from controlled simulations (like our Company Builder)

### X Threshold Selection
- **10%** (paper's choice): Balanced approach
- **5%**: More aggressive pruning (stricter)
- **15-20%**: More conservative (keeps more transitions)

### Expected Behavior
With synthetic data and no measurement error:
- ILM pruning should remove few/no workers
- Largest ILM should contain all or most jobs
- MVR should identify ground truth K correctly

With real-world data (e.g., LinkedIn):
- ILM pruning may remove 5-15% of workers
- Multiple ILMs may exist (we use largest)
- Results should be more robust to measurement error

## References

**Primary Paper**:
Huitfeldt, I., Kostøl, A. R., Vigtel, T. C., & Haller, A. (2023). "Internal labor markets: A worker flow approach." IZA Discussion Paper No. 14637.

**Key Sections**:
- Section 3.1: Identifying Internal Labor Markets
- Section 3.2: Job Ladders (MVR Algorithm)
- Appendix B.1: Algorithm 1 (ILM Pruning)
- Appendix B.2: Algorithm 2 (MVR)
- Appendix B.3: K-means Clustering

**Referenced Methods**:
- Clauset et al. (2015): Minimum violation ranking
- Bonhomme et al. (2019): K-means threshold for clustering
- Kline et al. (2020): Connected components approach
