"""
MVR Algorithm Variants Module

This module provides multiple implementations of the MVR (Minimum Violation Ranking) 
algorithm and related components, including:
- Paper-exact implementations (Huitfeldt et al., 2023)
- Alternative variants for robustness testing

Each function is clearly documented with references to specific sections of the paper.

References:
    Huitfeldt, I., Kostøl, A. R., Nimczik, J., & Weber, A. (2023). 
    Internal labor markets: A worker flow approach. IZA Discussion Paper No. 14637.
"""

import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Set
import random


# ==============================================================================
# STEP 1: BIPARTITE GRAPH CONSTRUCTION
# ==============================================================================

def build_bipartite_graph(panel_df: pd.DataFrame) -> nx.Graph:
    """
    Build bipartite graph G connecting workers and occupations.
    
    Paper Reference: Section 3.1, Algorithm 1 (Appendix B.1)
    "Let G = (U, V, E) denote a firm-specific bi-partite graph where U denotes 
    the set of workers in the firm, V denotes the set of occupations..."
    
    Args:
        panel_df: DataFrame with columns ['worker_id', 'year', 'role']
    
    Returns:
        Bipartite graph with workers (bipartite=0) and jobs (bipartite=1)
    """
    G = nx.Graph()
    
    for worker, group in panel_df.groupby('worker_id'):
        path = group.sort_values('year')['role'].tolist()
        
        # Remove consecutive duplicates (worker stays in same role)
        path_unique = [path[0]]
        for role in path[1:]:
            if role != path_unique[-1]:
                path_unique.append(role)
        
        # Add worker node and connect to all jobs in their path
        G.add_node(worker, bipartite=0)
        for role in path_unique:
            if role not in G:
                G.add_node(role, bipartite=1)
            G.add_edge(worker, role)
    
    return G


# ==============================================================================
# STEP 2: ILM NETWORK PRUNING (Algorithm 1)
# ==============================================================================

def prune_ilm_network(G: nx.Graph, X: int = 10) -> nx.Graph:
    """
    Apply leave-X-percent-out pruning to remove measurement error.
    
    Paper Reference: Algorithm 1 (Appendix B.1)
    "We apply a pruning algorithm related to the method used by Kline et al. (2020),
    which checks if removing a single worker breaks an ILM into further sub-markets."
    
    Algorithm:
    1. Compute degree d_v for each occupation v
    2. Construct G' where each link entering occupation v is duplicated 100/(d_v * X) times
    3. Identify articulation points in G' that are workers
    4. Remove those workers from G
    
    Args:
        G: Bipartite graph (workers and jobs)
        X: Threshold percentage (default 10%)
    
    Returns:
        Pruned graph with problematic workers removed
    """
    # Get job nodes and compute their degrees
    job_nodes = {n for n, d in G.nodes(data=True) if d.get('bipartite') == 1}
    job_degrees = {job: G.degree(job) for job in job_nodes}
    
    # Create G' with duplicated edges (paper's step 2)
    G_prime = nx.Graph()
    
    for u, v in G.edges():
        # If v is a job node, duplicate based on formula 100/(d_v * X)
        if v in job_nodes and job_degrees[v] > 0:
            duplication_factor = int(100 / (job_degrees[v] * X))
            for _ in range(duplication_factor):
                G_prime.add_edge(u, v)
        elif u in job_nodes and job_degrees[u] > 0:
            duplication_factor = int(100 / (job_degrees[u] * X))
            for _ in range(duplication_factor):
                G_prime.add_edge(u, v)
        else:
            G_prime.add_edge(u, v)
    
    # Copy node attributes
    for node, data in G.nodes(data=True):
        if node not in G_prime:
            G_prime.add_node(node, **data)
        else:
            for key, value in data.items():
                G_prime.nodes[node][key] = value
    
    # Find articulation points (cut vertices) in G'
    articulation_points = set(nx.articulation_points(G_prime))
    
    # Filter to only worker articulation points
    worker_nodes = {n for n, d in G.nodes(data=True) if d.get('bipartite') == 0}
    workers_to_remove = articulation_points & worker_nodes
    
    # Remove from original graph G (paper's step 4)
    G_pruned = G.copy()
    G_pruned.remove_nodes_from(workers_to_remove)
    
    return G_pruned


# ==============================================================================
# STEP 3: EXTRACT LARGEST CONNECTED COMPONENT (Largest ILM)
# ==============================================================================

def get_largest_connected_component(G: nx.Graph) -> nx.Graph:
    """
    Extract largest connected component from bipartite graph.
    
    Paper Reference: Section 3.1
    "We interpret these components as ILMs... focusing on the largest ILM of the firm."
    
    Args:
        G: Bipartite graph (possibly pruned)
    
    Returns:
        Subgraph containing the largest connected component (largest ILM)
    """
    if len(G.nodes()) == 0:
        return nx.Graph()
    
    components = list(nx.connected_components(G))
    
    if components:
        largest_component = max(components, key=len)
        return G.subgraph(largest_component).copy()
    
    return nx.Graph()


# ==============================================================================
# STEP 4: DIRECTED GRAPH CONSTRUCTION
# ==============================================================================

def build_directed_graph_consecutive(panel_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build directed graph H using consecutive transitions only (PAPER-EXACT METHOD).
    
    Paper Reference: Section 3.2
    "observed worker flows between occupations" - implies actual transitions,
    not all possible pairs in a career path.
    
    Theory: A promotion from A to C should be decomposed as A->B->C if B was 
    an intermediate step, not treated as a direct A->C edge.
    
    Example:
        Worker path: [Sales_L1, Sales_L2, Sales_L3]
        Creates edges: Sales_L1 -> Sales_L2, Sales_L2 -> Sales_L3
    
    Args:
        panel_df: DataFrame with columns ['worker_id', 'year', 'role']
    
    Returns:
        Directed graph with weighted edges (weight = transition frequency)
    """
    H = nx.DiGraph()
    
    for worker, group in panel_df.groupby('worker_id'):
        path = group.sort_values('year')['role'].tolist()
        
        # Remove consecutive duplicates
        path_unique = [path[0]]
        for role in path[1:]:
            if role != path_unique[-1]:
                path_unique.append(role)
        
        # Create consecutive edges only
        for i in range(len(path_unique) - 1):
            u, v = path_unique[i], path_unique[i + 1]
            if H.has_edge(u, v):
                H[u][v]['weight'] += 1
            else:
                H.add_edge(u, v, weight=1)
    
    return H


def build_directed_graph_all_pairs(panel_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build directed graph H using all pairs method (ALTERNATIVE METHOD).
    
    Note: This is NOT the paper method. Used for comparison.
    Creates transitive edges that may not represent actual transitions.
    
    Theory: If a worker goes A->B->C, this creates edges A->B, A->C, B->C.
    The A->C edge represents a "two-step promotion" path.
    
    Example:
        Worker path: [Sales_L1, Sales_L2, Sales_L3]
        Creates edges: Sales_L1 -> Sales_L2, Sales_L1 -> Sales_L3, Sales_L2 -> Sales_L3
    
    Args:
        panel_df: DataFrame with columns ['worker_id', 'year', 'role']
    
    Returns:
        Directed graph with weighted edges
    """
    H = nx.DiGraph()
    
    for worker, group in panel_df.groupby('worker_id'):
        path = group.sort_values('year')['role'].tolist()
        
        path_unique = [path[0]]
        for role in path[1:]:
            if role != path_unique[-1]:
                path_unique.append(role)
        
        # All pairs
        for i in range(len(path_unique)):
            for j in range(i + 1, len(path_unique)):
                u, v = path_unique[i], path_unique[j]
                if H.has_edge(u, v):
                    H[u][v]['weight'] += 1
                else:
                    H.add_edge(u, v, weight=1)
    
    return H


# ==============================================================================
# STEP 5: MINIMUM VIOLATION RANKING (Algorithm 2)
# ==============================================================================

def get_initial_ranking_unweighted(H: nx.DiGraph) -> List[str]:
    """
    Create initial ranking by unweighted out-degree (PAPER-EXACT METHOD).
    
    Paper Reference: Algorithm 2 (Appendix B.2)
    "sort occupations according to out-degree (decreasing)"
    
    Args:
        H: Directed graph of job transitions
    
    Returns:
        List of jobs sorted by out-degree (descending)
    """
    jobs = list(H.nodes())
    jobs.sort(key=lambda x: H.out_degree(x), reverse=True)
    return jobs


def get_initial_ranking_weighted(H: nx.DiGraph) -> List[str]:
    """
    Create initial ranking by weighted out-degree (ALTERNATIVE METHOD).
    
    Note: This is NOT the paper method. Used for comparison.
    Considers edge weights (transition frequencies).
    
    Args:
        H: Directed graph of job transitions
    
    Returns:
        List of jobs sorted by weighted out-degree (descending)
    """
    jobs = list(H.nodes())
    jobs.sort(key=lambda x: H.out_degree(x, weight='weight'), reverse=True)
    return jobs


def compute_violations(H: nx.DiGraph, ranking: List[str]) -> int:
    """
    Count number of violations in a given ranking (PAPER-EXACT METHOD).
    
    Paper Reference: Algorithm 2 (Appendix B.2)
    "A violation is a worker transition (u, v) where the rank of the origin
    occupation u is higher than the rank of the target occupation v."
    
    Note: Violations are counted UNWEIGHTED (each edge counts as 1).
    
    Args:
        H: Directed graph
        ranking: List of jobs in ranked order
    
    Returns:
        Number of violations (edges going "down" the ranking)
    """
    rank_dict = {job: i for i, job in enumerate(ranking)}
    violations = sum(
        1  # Count edges, not weights
        for u, v in H.edges()
        if rank_dict[u] > rank_dict[v]
    )
    return violations


def find_optimal_rankings_mvr(
    H: nx.DiGraph, 
    R: int, 
    T: int, 
    initial_ranking_method: str = "unweighted",
    seed: int = 42,
    progress_callback=None
) -> Tuple[List[Tuple[str]], int, List[int]]:
    """
    Run MVR algorithm to find optimal rankings (PAPER-EXACT METHOD).
    
    Paper Reference: Algorithm 2 (Appendix B.2)
    
    Algorithm:
    1. Initialize with ranking sorted by out-degree
    2. For R repetitions:
        a. For T iterations:
            - Randomly swap two jobs
            - Accept if violations <= current violations
        b. Track optimal rankings with minimum violations
    3. Return all optimal rankings
    
    Args:
        H: Directed graph
        R: Number of repetitions (paper uses 3000)
        T: Number of iterations per repetition (paper uses 1500)
        initial_ranking_method: "unweighted" or "weighted"
        seed: Random seed for reproducibility
        progress_callback: Optional callback function(current_rep, total_reps)
    
    Returns:
        Tuple of (optimal_rankings, min_violations, progress_history)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Get initial ranking
    if initial_ranking_method == "unweighted":
        initial_ranking = get_initial_ranking_unweighted(H)
    else:
        initial_ranking = get_initial_ranking_weighted(H)
    
    jobs = list(H.nodes())
    n = len(jobs)
    
    if n <= 1:
        return ([tuple(jobs)], 0, [1])
    
    current_ranking = initial_ranking.copy()
    current_violations = compute_violations(H, current_ranking)
    
    min_violations = current_violations
    optimal_rankings = set()
    optimal_rankings.add(tuple(current_ranking))
    progress = []
    
    for r in range(R):
        for t in range(T):
            # Random swap
            i, j = random.sample(range(n), 2)
            new_ranking = current_ranking.copy()
            new_ranking[i], new_ranking[j] = new_ranking[j], new_ranking[i]
            
            new_violations = compute_violations(H, new_ranking)
            
            # Accept if better or equal (paper: "accepts swaps where... S' <= S")
            if new_violations <= current_violations:
                current_ranking = new_ranking
                current_violations = new_violations
                
                if new_violations < min_violations:
                    min_violations = new_violations
                    optimal_rankings.clear()
                    optimal_rankings.add(tuple(current_ranking))
                elif new_violations == min_violations:
                    optimal_rankings.add(tuple(current_ranking))
        
        progress.append(len(optimal_rankings))
        
        if progress_callback:
            progress_callback(r + 1, R)
    
    return list(optimal_rankings), min_violations, progress


# ==============================================================================
# STEP 6: K-MEANS CLUSTERING (Appendix B.3)
# ==============================================================================

def compute_kmeans_threshold_bonhomme(
    positions: Dict[str, List[int]], 
    jobs: List[str],
    scaled: bool = False
) -> float:
    """
    Compute K-means threshold using Bonhomme et al. (2019) method.
    
    Paper Reference: Appendix B.3, Equation B2
    
    Paper-exact formula:
        threshold = (1/(N-1)) * sum(Var(r_v))
    
    where Var(r_v) is variance of job v's position across optimal rankings.
    
    Args:
        positions: Dictionary mapping job to list of positions in optimal rankings
        jobs: List of job names
        scaled: If True, multiply by N (alternative version, NOT paper-exact)
    
    Returns:
        Threshold value for K-means
    """
    N = len(jobs)
    sum_var = sum(
        np.var(positions[j], ddof=1) if len(positions[j]) > 1 else 0.0
        for j in jobs
    )
    
    if scaled:
        # Alternative (current implementation): multiply by N
        threshold = (1.0 / (N - 1)) * sum_var * N
    else:
        # Paper-exact: no multiplication by N
        threshold = sum_var / (N - 1)
    
    return threshold


def compute_kmeans_threshold_overall_variance(
    positions: Dict[str, List[int]], 
    jobs: List[str],
    scaled: bool = False
) -> float:
    """
    Compute K-means threshold using overall variance of consensus ranks.
    
    Note: This is NOT the paper method. Alternative for comparison.
    
    Uses variance of the average ranks instead of sum of individual variances.
    
    Args:
        positions: Dictionary mapping job to list of positions
        jobs: List of job names
        scaled: If True, multiply by N
    
    Returns:
        Threshold value
    """
    N = len(jobs)
    average_ranks = {job: np.mean(positions[job]) for job in jobs}
    rank_var = np.var(list(average_ranks.values()))
    
    if scaled:
        threshold = (1.0 / (N - 1)) * rank_var * N
    else:
        threshold = rank_var / (N - 1)
    
    return threshold


def run_kmeans_clustering(
    optimal_rankings: List[Tuple[str]],
    threshold_method: str = "bonhomme_exact"
) -> Dict:
    """
    Run K-means clustering on consensus rankings to identify hierarchy levels.
    
    Paper Reference: Appendix B.3
    
    Algorithm:
    1. Compute positions of each job across all optimal rankings
    2. Calculate threshold based on variance
    3. Find smallest K where Q(K) <= threshold
    4. Apply K-means with optimal K
    5. Factorize labels to get consecutive layer numbers (0, 1, 2, ...)
    
    Args:
        optimal_rankings: List of optimal rankings from MVR
        threshold_method: One of:
            - "bonhomme_exact": Paper-exact method
            - "bonhomme_scaled": Current implementation (scaled)
            - "overall_exact": Overall variance (unscaled)
            - "overall_scaled": Overall variance (scaled)
    
    Returns:
        Dictionary containing:
            - K: Optimal number of layers
            - threshold: Threshold value used
            - Q_values: List of Q(K) for each K
            - positions: Position of each job in optimal rankings
            - labels: Dictionary mapping job to layer number (factorized)
    """
    # Extract positions
    positions = defaultdict(list)
    for ranking in optimal_rankings:
        for pos, job in enumerate(ranking):
            positions[job].append(pos)
    
    jobs = sorted(positions.keys(), key=lambda x: np.mean(positions[x]))
    N = len(jobs)
    
    if N == 1:
        return {
            'K': 1,
            'threshold': 0.0,
            'Q_values': [0.0],
            'positions': positions,
            'labels': {jobs[0]: 0}
        }
    
    # Compute threshold based on method
    if threshold_method == "bonhomme_exact":
        threshold = compute_kmeans_threshold_bonhomme(positions, jobs, scaled=False)
    elif threshold_method == "bonhomme_scaled":
        threshold = compute_kmeans_threshold_bonhomme(positions, jobs, scaled=True)
    elif threshold_method == "overall_exact":
        threshold = compute_kmeans_threshold_overall_variance(positions, jobs, scaled=False)
    elif threshold_method == "overall_scaled":
        threshold = compute_kmeans_threshold_overall_variance(positions, jobs, scaled=True)
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")
    
    # Compute Q(K) for all K
    Q_values = []
    optimal_K = N
    
    for K in range(1, N + 1):
        if K == N:
            Q_K = 0.0
        else:
            kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
            job_mean_ranks = np.array([np.mean(positions[j]) for j in jobs]).reshape(-1, 1)
            kmeans.fit(job_mean_ranks)
            Q_K = kmeans.inertia_
        
        Q_values.append(Q_K)
        
        # Find first K where Q(K) <= threshold
        if Q_K <= threshold and optimal_K == N:
            optimal_K = K
    
    # Apply K-means with optimal K and factorize labels
    if optimal_K == N:
        labels_factorized = {jobs[i]: i for i in range(len(jobs))}
    else:
        kmeans_final = KMeans(n_clusters=optimal_K, random_state=0, n_init=10)
        job_mean_ranks = np.array([np.mean(positions[j]) for j in jobs]).reshape(-1, 1)
        labels = kmeans_final.fit_predict(job_mean_ranks)
        
        # Factorize: ensure labels are consecutive 0, 1, 2, ...
        labels_sorted = [labels[jobs.index(j)] for j in jobs]
        layer_ids = pd.factorize(labels_sorted)[0]
        labels_factorized = {jobs[i]: layer_ids[i] for i in range(len(jobs))}
    
    return {
        'K': optimal_K,
        'threshold': threshold,
        'Q_values': Q_values,
        'positions': positions,
        'labels': labels_factorized
    }


# ==============================================================================
# COMPLETE PIPELINE
# ==============================================================================

def run_complete_mvr_pipeline(
    panel_df: pd.DataFrame,
    enable_ilm_pruning: bool = True,
    X_threshold: int = 10,
    graph_method: str = "consecutive",
    ranking_method: str = "unweighted",
    threshold_method: str = "bonhomme_exact",
    R: int = 1000,
    T: int = 1000,
    seed: int = 42,
    progress_callback=None
) -> Dict:
    """
    Run complete MVR pipeline from raw panel data to hierarchy levels.
    
    This implements the full paper methodology:
    1. Build bipartite graph
    2. (Optional) Prune ILM network
    3. Extract largest connected component
    4. Build directed graph
    5. Run MVR algorithm
    6. K-means clustering
    
    Args:
        panel_df: Worker trajectories with ['worker_id', 'year', 'role']
        enable_ilm_pruning: Whether to apply Algorithm 1 pruning
        X_threshold: Threshold for pruning (default 10%)
        graph_method: "consecutive" or "all_pairs"
        ranking_method: "unweighted" or "weighted"
        threshold_method: K-means threshold method
        R: MVR repetitions
        T: MVR iterations
        seed: Random seed
        progress_callback: Optional progress callback
    
    Returns:
        Dictionary with all results
    """
    results = {}
    
    # Step 1: Build bipartite graph
    G = build_bipartite_graph(panel_df)
    results['bipartite_graph'] = G
    results['initial_jobs'] = len([n for n, d in G.nodes(data=True) if d.get('bipartite') == 1])
    results['initial_workers'] = len([n for n, d in G.nodes(data=True) if d.get('bipartite') == 0])
    
    # Step 2 & 3: Optionally prune and get largest ILM
    if enable_ilm_pruning:
        G_pruned = prune_ilm_network(G, X=X_threshold)
        workers_removed = results['initial_workers'] - len([n for n, d in G_pruned.nodes(data=True) if d.get('bipartite') == 0])
        results['workers_removed'] = workers_removed
        G_ilm = get_largest_connected_component(G_pruned)
    else:
        G_ilm = get_largest_connected_component(G)
        results['workers_removed'] = 0
    
    results['largest_ilm_graph'] = G_ilm
    results['largest_ilm_jobs'] = len([n for n, d in G_ilm.nodes(data=True) if d.get('bipartite') == 1])
    results['largest_ilm_workers'] = len([n for n, d in G_ilm.nodes(data=True) if d.get('bipartite') == 0])
    
    # Filter panel_df to only include workers/jobs in largest ILM
    ilm_workers = {n for n, d in G_ilm.nodes(data=True) if d.get('bipartite') == 0}
    ilm_jobs = {n for n, d in G_ilm.nodes(data=True) if d.get('bipartite') == 1}
    panel_filtered = panel_df[
        (panel_df['worker_id'].isin(ilm_workers)) &
        (panel_df['role'].isin(ilm_jobs))
    ]
    
    # Step 4: Build directed graph
    if graph_method == "consecutive":
        H = build_directed_graph_consecutive(panel_filtered)
    else:
        H = build_directed_graph_all_pairs(panel_filtered)
    
    results['directed_graph'] = H
    results['directed_graph_nodes'] = H.number_of_nodes()
    results['directed_graph_edges'] = H.number_of_edges()
    
    # Step 5: Run MVR
    optimal_rankings, min_violations, progress = find_optimal_rankings_mvr(
        H, R, T, 
        initial_ranking_method=ranking_method,
        seed=seed,
        progress_callback=progress_callback
    )
    
    results['optimal_rankings'] = optimal_rankings
    results['min_violations'] = min_violations
    results['progress'] = progress
    
    # Step 6: K-means clustering
    kmeans_result = run_kmeans_clustering(optimal_rankings, threshold_method)
    results.update(kmeans_result)
    
    return results
