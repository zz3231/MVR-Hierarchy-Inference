"""
Unit tests for MVR algorithms.

Run with: python test_mvr_algorithms.py
"""

import pandas as pd
import networkx as nx
import numpy as np
from mvr_algorithms import (
    build_bipartite_graph,
    build_directed_graph_consecutive,
    build_directed_graph_all_pairs,
    get_initial_ranking_unweighted,
    get_initial_ranking_weighted,
    compute_violations,
    find_optimal_rankings_mvr,
    run_kmeans_clustering
)


def test_simple_linear_path():
    """
    Test 1: Simple linear path A -> B -> C
    
    Expected:
    - Consecutive creates 2 edges: A->B, B->C
    - All pairs creates 3 edges: A->B, A->C, B->C
    - Optimal ranking: [A, B, C]
    - Violations: 0
    """
    print("\n" + "="*60)
    print("TEST 1: Simple Linear Path (A -> B -> C)")
    print("="*60)
    
    # Create simple panel data
    panel_df = pd.DataFrame([
        {'worker_id': 1, 'year': 2020, 'role': 'A'},
        {'worker_id': 1, 'year': 2021, 'role': 'B'},
        {'worker_id': 1, 'year': 2022, 'role': 'C'},
        
        {'worker_id': 2, 'year': 2020, 'role': 'A'},
        {'worker_id': 2, 'year': 2021, 'role': 'B'},
        {'worker_id': 2, 'year': 2022, 'role': 'C'},
    ])
    
    # Test consecutive method
    H_consecutive = build_directed_graph_consecutive(panel_df)
    print(f"\nConsecutive method:")
    print(f"  Nodes: {list(H_consecutive.nodes())}")
    print(f"  Edges: {list(H_consecutive.edges())}")
    print(f"  Expected: A->B, B->C")
    assert H_consecutive.number_of_edges() == 2, "Should have 2 edges"
    assert H_consecutive.has_edge('A', 'B'), "Should have A->B"
    assert H_consecutive.has_edge('B', 'C'), "Should have B->C"
    assert not H_consecutive.has_edge('A', 'C'), "Should NOT have A->C"
    print("  PASS")
    
    # Test all pairs method
    H_all_pairs = build_directed_graph_all_pairs(panel_df)
    print(f"\nAll pairs method:")
    print(f"  Nodes: {list(H_all_pairs.nodes())}")
    print(f"  Edges: {list(H_all_pairs.edges())}")
    print(f"  Expected: A->B, A->C, B->C")
    assert H_all_pairs.number_of_edges() == 3, "Should have 3 edges"
    assert H_all_pairs.has_edge('A', 'B'), "Should have A->B"
    assert H_all_pairs.has_edge('B', 'C'), "Should have B->C"
    assert H_all_pairs.has_edge('A', 'C'), "Should have A->C"
    print("  PASS")
    
    # Test ranking
    ranking_unweighted = get_initial_ranking_unweighted(H_consecutive)
    print(f"\nInitial ranking (unweighted): {ranking_unweighted}")
    print(f"  Expected: ['A', 'B', 'C'] (A has highest out-degree)")
    
    # Test violations
    optimal_ranking = ['A', 'B', 'C']
    violations = compute_violations(H_consecutive, optimal_ranking)
    print(f"\nViolations for correct ranking: {violations}")
    print(f"  Expected: 0")
    assert violations == 0, "Correct ranking should have 0 violations"
    
    wrong_ranking = ['C', 'B', 'A']
    violations_wrong = compute_violations(H_consecutive, wrong_ranking)
    print(f"Violations for reversed ranking: {violations_wrong}")
    print(f"  Expected: 2 (both edges violated)")
    assert violations_wrong == 2, "Reversed ranking should have 2 violations"
    
    print("\nTEST 1: PASSED")


def test_y_shaped_structure():
    """
    Test 2: Y-shaped structure
    
         B
        /
    A -
        \
         C
    
    Expected:
    - A should rank lowest
    - B and C are ambiguous (no direct connection)
    - Multiple optimal rankings possible: [A,B,C] and [A,C,B]
    """
    print("\n" + "="*60)
    print("TEST 2: Y-shaped Structure")
    print("="*60)
    
    panel_df = pd.DataFrame([
        # Worker 1: A -> B
        {'worker_id': 1, 'year': 2020, 'role': 'A'},
        {'worker_id': 1, 'year': 2021, 'role': 'B'},
        
        # Worker 2: A -> C
        {'worker_id': 2, 'year': 2020, 'role': 'A'},
        {'worker_id': 2, 'year': 2021, 'role': 'C'},
        
        # Worker 3: A -> B
        {'worker_id': 3, 'year': 2020, 'role': 'A'},
        {'worker_id': 3, 'year': 2021, 'role': 'B'},
    ])
    
    H = build_directed_graph_consecutive(panel_df)
    print(f"\nEdges: {list(H.edges())}")
    print(f"Expected: A->B (weight=2), A->C (weight=1)")
    
    # Both [A,B,C] and [A,C,B] should have 0 violations
    ranking1 = ['A', 'B', 'C']
    ranking2 = ['A', 'C', 'B']
    
    viol1 = compute_violations(H, ranking1)
    viol2 = compute_violations(H, ranking2)
    
    print(f"\nViolations for [A,B,C]: {viol1}")
    print(f"Violations for [A,C,B]: {viol2}")
    print(f"Expected: Both should be 0")
    
    assert viol1 == 0, "Both rankings should be valid"
    assert viol2 == 0, "Both rankings should be valid"
    
    print("\nTEST 2: PASSED")


def test_cycle_structure():
    """
    Test 3: Cycle structure (should have violations)
    
    A -> B -> C -> A
    
    Expected:
    - No perfect ranking (all rankings have >= 1 violation)
    - MVR should find ranking with minimum violations
    """
    print("\n" + "="*60)
    print("TEST 3: Cycle Structure")
    print("="*60)
    
    panel_df = pd.DataFrame([
        # Worker 1: A -> B
        {'worker_id': 1, 'year': 2020, 'role': 'A'},
        {'worker_id': 1, 'year': 2021, 'role': 'B'},
        
        # Worker 2: B -> C
        {'worker_id': 2, 'year': 2020, 'role': 'B'},
        {'worker_id': 2, 'year': 2021, 'role': 'C'},
        
        # Worker 3: C -> A (creates cycle)
        {'worker_id': 3, 'year': 2020, 'role': 'C'},
        {'worker_id': 3, 'year': 2021, 'role': 'A'},
    ])
    
    H = build_directed_graph_consecutive(panel_df)
    print(f"\nEdges: {list(H.edges())}")
    print(f"Expected: A->B, B->C, C->A (forms cycle)")
    
    # Test all possible rankings
    rankings = [
        ['A', 'B', 'C'],
        ['A', 'C', 'B'],
        ['B', 'A', 'C'],
        ['B', 'C', 'A'],
        ['C', 'A', 'B'],
        ['C', 'B', 'A']
    ]
    
    print("\nViolations for all possible rankings:")
    for ranking in rankings:
        viols = compute_violations(H, ranking)
        print(f"  {ranking}: {viols} violations")
    
    # All should have at least 1 violation due to cycle
    violations_list = [compute_violations(H, r) for r in rankings]
    min_viols = min(violations_list)
    
    print(f"\nMinimum violations possible: {min_viols}")
    print(f"Expected: 1 (cycle requires breaking at least one edge)")
    
    assert min_viols >= 1, "Cycle should require at least 1 violation"
    
    print("\nTEST 3: PASSED")


def test_threshold_calculation():
    """
    Test 4: Verify threshold calculations match paper formulas.
    """
    print("\n" + "="*60)
    print("TEST 4: Threshold Calculation")
    print("="*60)
    
    # Create mock positions data
    positions = {
        'A': [0, 0, 0, 1],  # Var = 0.25
        'B': [1, 2, 1, 2],  # Var = 0.33
        'C': [3, 3, 3, 3]   # Var = 0
    }
    
    jobs = ['A', 'B', 'C']
    N = 3
    
    # Calculate variances manually
    var_A = np.var([0, 0, 0, 1], ddof=1)
    var_B = np.var([1, 2, 1, 2], ddof=1)
    var_C = 0.0
    
    sum_var = var_A + var_B + var_C
    
    print(f"\nVariances: A={var_A:.4f}, B={var_B:.4f}, C={var_C:.4f}")
    print(f"Sum of variances: {sum_var:.4f}")
    
    # Test Bonhomme exact (paper formula)
    from mvr_algorithms import compute_kmeans_threshold_bonhomme
    
    threshold_exact = compute_kmeans_threshold_bonhomme(positions, jobs, scaled=False)
    expected_exact = sum_var / (N - 1)
    
    print(f"\nBonhomme Exact:")
    print(f"  Calculated: {threshold_exact:.4f}")
    print(f"  Expected: {expected_exact:.4f}")
    print(f"  Formula: sum_var / (N-1) = {sum_var:.4f} / {N-1} = {expected_exact:.4f}")
    
    assert abs(threshold_exact - expected_exact) < 1e-6, "Threshold calculation mismatch"
    
    # Test Bonhomme scaled
    threshold_scaled = compute_kmeans_threshold_bonhomme(positions, jobs, scaled=True)
    expected_scaled = sum_var * N / (N - 1)
    
    print(f"\nBonhomme Scaled:")
    print(f"  Calculated: {threshold_scaled:.4f}")
    print(f"  Expected: {expected_scaled:.4f}")
    print(f"  Formula: sum_var * N / (N-1) = {sum_var:.4f} * {N} / {N-1} = {expected_scaled:.4f}")
    
    assert abs(threshold_scaled - expected_scaled) < 1e-6, "Threshold calculation mismatch"
    
    # Verify scaling relationship
    print(f"\nScaling factor: {threshold_scaled / threshold_exact:.4f}")
    print(f"Expected: {N} (N multiplier)")
    
    print("\nTEST 4: PASSED")


def test_consecutive_vs_all_pairs_comparison():
    """
    Test 5: Compare consecutive vs all pairs on realistic firm.
    """
    print("\n" + "="*60)
    print("TEST 5: Consecutive vs All Pairs Comparison")
    print("="*60)
    
    # Create firm with clear promotion paths
    panel_df = pd.DataFrame([
        # 10 workers: Junior -> Senior -> Manager
        *[{'worker_id': i, 'year': 2020, 'role': 'Junior'} for i in range(1, 11)],
        *[{'worker_id': i, 'year': 2021, 'role': 'Senior'} for i in range(1, 11)],
        *[{'worker_id': i, 'year': 2022, 'role': 'Manager'} for i in range(1, 11)],
        
        # 5 more workers: Junior -> Senior (stop at Senior)
        *[{'worker_id': i, 'year': 2020, 'role': 'Junior'} for i in range(11, 16)],
        *[{'worker_id': i, 'year': 2021, 'role': 'Senior'} for i in range(11, 16)],
    ])
    
    H_consecutive = build_directed_graph_consecutive(panel_df)
    H_all_pairs = build_directed_graph_all_pairs(panel_df)
    
    print(f"\nConsecutive method:")
    print(f"  Edges: {list(H_consecutive.edges(data=True))}")
    print(f"  Total edges: {H_consecutive.number_of_edges()}")
    
    print(f"\nAll pairs method:")
    print(f"  Edges: {list(H_all_pairs.edges(data=True))}")
    print(f"  Total edges: {H_all_pairs.number_of_edges()}")
    
    # Compare edge counts
    print(f"\nComparison:")
    print(f"  Consecutive: {H_consecutive.number_of_edges()} edges")
    print(f"  All pairs: {H_all_pairs.number_of_edges()} edges")
    print(f"  Difference: {H_all_pairs.number_of_edges() - H_consecutive.number_of_edges()} extra edges")
    
    # For this path, consecutive should have 2 edges, all pairs should have 3
    assert H_consecutive.number_of_edges() == 2, "Consecutive should have 2 edges"
    assert H_all_pairs.number_of_edges() == 3, "All pairs should have 3 edges"
    
    # Both should produce correct ranking [Junior, Senior, Manager]
    ranking = ['Junior', 'Senior', 'Manager']
    
    viol_consecutive = compute_violations(H_consecutive, ranking)
    viol_all_pairs = compute_violations(H_all_pairs, ranking)
    
    print(f"\nViolations for correct ranking ['Junior', 'Senior', 'Manager']:")
    print(f"  Consecutive: {viol_consecutive}")
    print(f"  All pairs: {viol_all_pairs}")
    print(f"  Expected: 0 for both")
    
    assert viol_consecutive == 0, "Correct ranking should have 0 violations"
    assert viol_all_pairs == 0, "Correct ranking should have 0 violations"
    
    print("\nTEST 5: PASSED")


def test_bipartite_graph():
    """
    Test 6: Bipartite graph construction
    """
    print("\n" + "="*60)
    print("TEST 6: Bipartite Graph Construction")
    print("="*60)
    
    panel_df = pd.DataFrame([
        {'worker_id': 1, 'year': 2020, 'role': 'A'},
        {'worker_id': 1, 'year': 2021, 'role': 'A'},  # Same role (should be deduplicated)
        {'worker_id': 1, 'year': 2022, 'role': 'B'},
        
        {'worker_id': 2, 'year': 2020, 'role': 'A'},
        {'worker_id': 2, 'year': 2021, 'role': 'C'},
    ])
    
    G = build_bipartite_graph(panel_df)
    
    worker_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
    job_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]
    
    print(f"\nWorker nodes: {worker_nodes}")
    print(f"Job nodes: {job_nodes}")
    
    print(f"\nEdges:")
    for u, v in G.edges():
        print(f"  {u} <-> {v}")
    
    assert len(worker_nodes) == 2, "Should have 2 workers"
    assert len(job_nodes) == 3, "Should have 3 jobs (A, B, C)"
    
    # Worker 1 should connect to A and B (not two As)
    worker1_jobs = list(G.neighbors(1))
    print(f"\nWorker 1 connected to: {worker1_jobs}")
    print(f"Expected: ['A', 'B'] (duplicate A removed)")
    assert set(worker1_jobs) == {'A', 'B'}, "Worker 1 should connect to A and B"
    
    # Worker 2 should connect to A and C
    worker2_jobs = list(G.neighbors(2))
    print(f"Worker 2 connected to: {worker2_jobs}")
    print(f"Expected: ['A', 'C']")
    assert set(worker2_jobs) == {'A', 'C'}, "Worker 2 should connect to A and C"
    
    print("\nTEST 6: PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING MVR ALGORITHM UNIT TESTS")
    print("="*60)
    
    try:
        test_simple_linear_path()
        test_y_shaped_structure()
        test_cycle_structure()
        test_threshold_calculation()
        test_consecutive_vs_all_pairs_comparison()
        test_bipartite_graph()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n\nTEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n\nERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
