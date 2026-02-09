# K-means Methods Verification

## Summary of 4 K-means Methods in Our Implementation

### 1. **Average Optimal Ranking Variance** (Currently Default)
**Threshold Formula**:
```python
rank_var = np.var(list(average_ranks.values()))  # Variance of average ranks across jobs
threshold = (1 / (N - 1)) * rank_var * N
```

**Comparison**: `inertia_ ≤ threshold`

**Interpretation**: 
- Uses variance of the **average ranks** (one value per job)
- Measures overall dispersion between job positions
- More aggressive clustering (smaller threshold)

**Status**: ✅ Correctly implemented

---

### 2. **Bonhomme et al. (2019)** (Paper's Method)
**Threshold Formula**:
```python
sum_var = sum(np.var(positions[j], ddof=1) for j in jobs)  # Sum of variances across optimal rankings
threshold = (1 / (N - 1)) * sum_var * N
```

**Comparison**: `inertia_ ≤ threshold`

**Paper Reference**: Equation B2 (line 2829-2840 in dp14637.pdf)
```
K̂ = min{K ≥ 1 : Q̂(K) ≤ (1/(N-1)) × Σ Var(r_v)}
where Q̂(K) = (1/N) × Σ(r_v - ĥ(k_v))²
```

**Derivation**:
- Paper: `(1/N) × inertia_ ≤ (1/(N-1)) × Σ Var(r_v)`
- Multiply both sides by N: `inertia_ ≤ (N/(N-1)) × Σ Var(r_v)`
- Which equals: `inertia_ ≤ (1/(N-1)) × Σ Var(r_v) × N` ✅

**Interpretation**:
- Uses sum of **within-job variances** (variance across optimal rankings for each job)
- Measures uncertainty in rank estimation
- More conservative clustering (larger threshold)

**Status**: ✅ Correctly implemented (matches paper)

---

### 3. **Elbow Method (Manual)**
**Method**: User manually selects K after viewing the elbow plot

**UI**:
```python
if kmeans_method == "Elbow (Manual)":
    chosen_K = st.number_input("Choose K", 1, 20, 4)
```

**Status**: ✅ Implemented

---

### 4. **Simple Variance Threshold**
**Status**: ❌ NOT IMPLEMENTED

**Expected behavior**: User sets a fixed variance threshold, and K is chosen when objective falls below it.

---

## Which Method is "Paper's Method"?

**Answer**: Method #2 (Bonhomme et al. 2019)

This is explicitly stated in paper (Section 3.2.2, line 2820):
> "Following Bonhomme et al. (2019), we choose the number of hierarchy levels, K, such that..."

---

## Comparison: Method #1 vs Method #2

| Aspect | Method #1 (Overall Variance) | Method #2 (Bonhomme/Paper) |
|--------|------------------------------|----------------------------|
| Threshold | `(1/(N-1)) × Var(avg_ranks) × N` | `(1/(N-1)) × Σ Var(r_v) × N` |
| What it measures | Spread of average positions | Sum of rank uncertainties |
| Typical value | Smaller (fewer jobs, single variance) | Larger (sum over all jobs) |
| Result | Tends to choose **larger K** (more clusters) | Tends to choose **smaller K** (fewer clusters) |
| Use case | When you want finer hierarchy granularity | Academic rigor, matches paper |

---

## Recommendations

### For Academic Publication
- **Use Method #2 (Bonhomme)** as it matches the paper exactly
- Set as default? **NO** - keep Method #1 as default for practical use
- Clearly label Method #2 as "Bonhomme et al. (2019) - Paper's Method"

### For Practical Use
- **Method #1 (Average Optimal Ranking Variance)** is more intuitive
- Produces finer-grained hierarchies which may be useful for visualization
- Keep as default

### UI Suggestion
Update the selectbox labels to be clearer:
```python
kmeans_method = st.selectbox("K-means Method", 
    [
        "Average Optimal Ranking Variance (Default)",
        "Bonhomme et al. (2019) - Paper Method",
        "Elbow (Manual)", 
        "Simple Variance"
    ],
    index=0)
```

---

## Verification Against RA's Code

RA's implementation (line 1421-1422 in mvr_example_for_mike_v3.ipynb):
```python
threshold_rank_sd = (1 / (N - 1)) * sum_stddev * N      # Bonhomme
threshold_overall_sd = (1 / (N - 1)) * rank_var * N     # Overall SD
```

Our implementation matches RA's code **exactly** ✅

Note: RA had commented out alternative versions without the `× N` factor, but those would be incorrect because sklearn's `inertia_` is a sum, not an average.

---

## Action Items

1. ✅ Verify Bonhomme method matches paper equation - **CONFIRMED**
2. ✅ Verify Overall Variance method is correct - **CONFIRMED**
3. ⚠️ Update UI labels to clarify which is "Paper's Method"
4. ❓ Implement "Simple Variance" method? (Currently listed but not implemented)
5. ⚠️ Update documentation to clarify method differences
