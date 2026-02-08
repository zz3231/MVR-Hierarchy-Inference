"""
MVR Hierarchy Inference Tool - Multi-page Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
from collections import defaultdict
from math import comb
import warnings
warnings.filterwarnings('ignore')

from firm_structure import FirmStructure, PromotionConfig, WorkerGenerator, ObservationBiasSimulator

st.set_page_config(
    page_title="MVR Hierarchy Inference",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'ground_truth_df' not in st.session_state:
    st.session_state.ground_truth_df = None
if 'biased_df' not in st.session_state:
    st.session_state.biased_df = None
if 'firm_structure' not in st.session_state:
    st.session_state.firm_structure = None

# Page navigation
st.sidebar.title("MVR Hierarchy Inference")
page = st.sidebar.radio("Navigate", ["Company Builder", "MVR Analysis"])

# ==================== PAGE 1: COMPANY BUILDER ====================

if page == "Company Builder":
    st.title("Synthetic Company Builder")
    st.markdown("""
    Build synthetic companies with configurable selection bias to test MVR algorithm robustness.
    """)
    
    st.subheader("1. Configure Firm Structure")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_departments = st.number_input("Number of Departments", 1, 6, 4)
        
        departments = []
        st.markdown("**Department Configuration:**")
        for i in range(n_departments):
            dept_col1, dept_col2 = st.columns([2, 1])
            with dept_col1:
                dept_name = st.text_input(f"Dept {i+1} Name", 
                                         value=["Law", "Engineering", "Sales", "Finance", "Marketing", "HR"][i],
                                         key=f"dept_name_{i}")
            with dept_col2:
                n_ranks = st.number_input(f"Ranks", 1, 10, 5, key=f"dept_ranks_{i}")
            departments.append((dept_name, n_ranks))
    
    with col2:
        if departments:
            firm = FirmStructure(departments)
            st.session_state.firm_structure = firm
            
            fig = firm.visualize_structure()
            st.pyplot(fig)
            plt.close()
    
    st.subheader("2. Generate Ground Truth Company")
    
    col1, col2 = st.columns(2)
    with col1:
        n_workers = st.number_input("Total Workers", 100, 2000, 500, 50)
        default_promo_rate = st.slider("Default Promotion Rate", 0.0, 0.5, 0.20, 0.05)
    with col2:
        start_year = st.number_input("Start Year", 2000, 2020, 2016)
        end_year = st.number_input("End Year", 2020, 2030, 2023)
    
    if st.button("Generate Ground Truth Company", type="primary"):
        with st.spinner("Generating workers..."):
            promo_config = PromotionConfig(firm, default_prob=default_promo_rate)
            worker_gen = WorkerGenerator(firm, promo_config, n_workers, start_year, end_year)
            trajectories = worker_gen.generate_trajectories()
            st.session_state.ground_truth_df = trajectories
            
        st.success(f"Generated {len(trajectories)} records for {trajectories['worker_id'].nunique()} workers")
        
        summary = worker_gen.get_summary_table(trajectories)
        st.markdown("**Ground Truth Company Summary:**")
        st.dataframe(summary, use_container_width=True)
    
    if st.session_state.ground_truth_df is not None:
        st.subheader("3. Configure Observation Bias")
        
        st.markdown("""
        Simulate LinkedIn data limitations by setting observation rates per rank.
        Example: Lower ranks may have 50% observation rate, higher ranks 100%.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            bias_type = st.radio("Bias Type", ["Layer-based", "Custom per Role"])
            
            if bias_type == "Layer-based":
                st.markdown("**Observation Rate by Rank:**")
                rates_by_rank = {}
                max_rank = st.session_state.ground_truth_df['rank'].max()
                for rank in range(1, int(max_rank) + 1):
                    rate = st.slider(f"Rank {rank}", 0.0, 1.0, 
                                    0.5 if rank <= 2 else (0.7 if rank <= 4 else 1.0),
                                    0.05, key=f"rank_rate_{rank}")
                    rates_by_rank[rank] = rate
        
        with col2:
            if st.button("Apply Observation Bias", type="primary"):
                bias_sim = ObservationBiasSimulator(st.session_state.ground_truth_df)
                
                if bias_type == "Layer-based":
                    bias_sim.set_layer_based_rates(rates_by_rank)
                
                biased_df = bias_sim.apply_bias()
                st.session_state.biased_df = biased_df
                
                st.success(f"Generated biased sample: {len(biased_df)} records ({len(biased_df)/len(st.session_state.ground_truth_df)*100:.1f}%)")
                
                comparison = bias_sim.get_comparison_table(biased_df)
                st.markdown("**Comparison: Ground Truth vs Observed:**")
                st.dataframe(comparison[['department', 'role', 'rank', 'ground_truth_avg', 
                                        'observed_avg', 'observation_rate', 'actual_rate']], 
                            use_container_width=True)
        
        if st.session_state.biased_df is not None:
            st.info("Both companies ready. Go to 'MVR Analysis' to compare algorithm performance.")

# ==================== PAGE 2: MVR ANALYSIS ====================

elif page == "MVR Analysis":
    st.title("MVR Ranking Analysis")
    
    if st.session_state.ground_truth_df is None:
        st.warning("Please generate companies in 'Company Builder' first.")
        st.stop()
    
    st.markdown("""
    Run MVR algorithm on both ground truth and biased companies.
    Compare whether the algorithm correctly identifies hierarchy levels despite selection bias.
    """)
    
    # MVR Parameters
    st.subheader("MVR Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        R = st.slider("R (Repetitions)", 100, 5000, 1000, 100)
    with col2:
        T = st.slider("T (Iterations)", 100, 3000, 1000, 100)
    with col3:
        early_stop = st.checkbox("Enable Early Stopping", False)
    
    # K-means method
    kmeans_method = st.selectbox("K-means Method", 
                                 ["Bonhomme et al. (2019)", "Overall Std", 
                                  "Elbow (Manual)", "Simple Variance"])
    
    if kmeans_method == "Elbow (Manual)":
        chosen_K = st.number_input("Choose K", 1, 20, 4)
    
    # Import MVR functions from original code
    def build_directed_graph_all_pairs(panel_df):
        """Build directed graph H using ALL PAIRS method"""
        H = nx.DiGraph()
        
        for worker, group in panel_df.groupby('worker_id'):
            path = group.sort_values('year')['role'].tolist()
            
            for i in range(len(path)):
                for j in range(i + 1, len(path)):
                    if path[i] != path[j]:  # Skip self-loops
                        if H.has_edge(path[i], path[j]):
                            H[path[i]][path[j]]['weight'] += 1
                        else:
                            H.add_edge(path[i], path[j], weight=1)
        
        return H
    
    def compute_violations_fast(H, ranking):
        """Count violations efficiently"""
        rank_dict = {job: i for i, job in enumerate(ranking)}
        violations = sum(
            H[u][v]['weight']
            for u, v in H.edges()
            if rank_dict[u] > rank_dict[v]
        )
        return violations
    
    def find_optimal_rankings_mvr(H, R, T, early_stop, seed=42):
        """MVR algorithm"""
        random.seed(seed)
        np.random.seed(seed)
        
        jobs = list(H.nodes())
        n = len(jobs)
        jobs.sort(key=lambda x: H.out_degree(x, weight='weight'), reverse=True)
        current_ranking = jobs.copy()
        current_violations = compute_violations_fast(H, current_ranking)
        
        min_violations = current_violations
        optimal_rankings = set()
        optimal_rankings.add(tuple(current_ranking))
        progress = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for r in range(R):
            for t in range(T):
                i, j = random.sample(range(n), 2)
                new_ranking = current_ranking.copy()
                new_ranking[i], new_ranking[j] = new_ranking[j], new_ranking[i]
                new_violations = compute_violations_fast(H, new_ranking)
                
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
            progress_bar.progress((r + 1) / R)
            if (r + 1) % 100 == 0:
                status_text.text(f"Rep {r + 1}/{R}: {len(optimal_rankings)} rankings")
            
            if early_stop and r >= 500:
                if len(set(progress[-500:])) == 1:
                    status_text.text(f"Early stop at R={r+1}")
                    progress_bar.progress(1.0)
                    break
        
        progress_bar.empty()
        status_text.empty()
        
        return list(optimal_rankings), min_violations, progress
    
    def run_kmeans_bonhomme(optimal_rankings):
        """Bonhomme method"""
        positions = defaultdict(list)
        for ranking in optimal_rankings:
            for pos, job in enumerate(ranking):
                positions[job].append(pos)
        
        jobs = sorted(positions.keys(), key=lambda x: np.mean(positions[x]))
        N = len(jobs)
        sum_var = sum(np.var(positions[j], ddof=1) if len(positions[j]) > 1 else 0 for j in jobs)
        threshold = (1 / (N - 1)) * sum_var * N
        
        for K in range(1, N + 1):
            if K == N:
                Q_K = 0.0
            else:
                kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
                job_mean_ranks = np.array([np.mean(positions[j]) for j in jobs]).reshape(-1, 1)
                kmeans.fit(job_mean_ranks)
                Q_K = kmeans.inertia_
            
            if Q_K <= threshold:
                return K
        return N
    
    # Run analysis
    if st.button("Run MVR Analysis on Both Companies", type="primary"):
        st.subheader("Results Comparison")
        
        col1, col2 = st.columns(2)
        
        # Ground Truth Company
        with col1:
            st.markdown("**Ground Truth Company**")
            with st.spinner("Building graph..."):
                H_gt = build_directed_graph_all_pairs(st.session_state.ground_truth_df)
            st.write(f"Graph: {H_gt.number_of_nodes()} nodes, {H_gt.number_of_edges()} edges")
            
            with st.spinner("Running MVR..."):
                opt_gt, viol_gt, prog_gt = find_optimal_rankings_mvr(H_gt, R, T, early_stop)
            st.write(f"Found {len(opt_gt)} optimal rankings")
            st.write(f"Min violations: {viol_gt}")
            
            K_gt = run_kmeans_bonhomme(opt_gt)
            st.metric("Identified Layers (K)", K_gt)
        
        # Biased Company
        with col2:
            st.markdown("**Biased Company (Observed Data)**")
            with st.spinner("Building graph..."):
                H_bias = build_directed_graph_all_pairs(st.session_state.biased_df)
            st.write(f"Graph: {H_bias.number_of_nodes()} nodes, {H_bias.number_of_edges()} edges")
            
            with st.spinner("Running MVR..."):
                opt_bias, viol_bias, prog_bias = find_optimal_rankings_mvr(H_bias, R, T, early_stop)
            st.write(f"Found {len(opt_bias)} optimal rankings")
            st.write(f"Min violations: {viol_bias}")
            
            K_bias = run_kmeans_bonhomme(opt_bias)
            st.metric("Identified Layers (K)", K_bias)
        
        # Comparison
        st.subheader("Algorithm Robustness Assessment")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ground Truth K", K_gt)
        with col2:
            st.metric("Biased Data K", K_bias)
        with col3:
            diff = abs(K_gt - K_bias)
            st.metric("Difference", diff, delta=f"{diff} layers")
        
        if K_gt == K_bias:
            st.success("Algorithm correctly identified the same number of layers despite selection bias.")
        else:
            st.warning(f"Algorithm identified different layer counts: GT={K_gt}, Biased={K_bias}")
        
        # Convergence plots
        st.subheader("Convergence Comparison")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(range(1, len(prog_gt) + 1), prog_gt, 'b-', linewidth=2)
        ax1.set_title('Ground Truth Company')
        ax1.set_xlabel('Repetitions')
        ax1.set_ylabel('Unique Rankings Found')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(range(1, len(prog_bias) + 1), prog_bias, 'r-', linewidth=2)
        ax2.set_title('Biased Company')
        ax2.set_xlabel('Repetitions')
        ax2.set_ylabel('Unique Rankings Found')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

if __name__ == "__main__":
    pass
