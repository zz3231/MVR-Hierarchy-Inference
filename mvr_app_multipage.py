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
if 'mvr_results' not in st.session_state:
    st.session_state.mvr_results = None

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
        n_departments = st.number_input("Number of Departments", 1, 6, 3)
        
        departments = []
        st.markdown("**Department Configuration:**")
        for i in range(n_departments):
            dept_col1, dept_col2 = st.columns([2, 1])
            with dept_col1:
                dept_name = st.text_input(f"Dept {i+1} Name", 
                                         value=["Sales", "Engineering", "Finance", "Law", "Marketing", "HR"][i],
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
        n_workers = st.number_input("Total Workers", 100, 2000, 1000, 50)
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
        Set observation rate for each role individually.
        Rate = 1.0 means 100% observed, 0.5 means 50% observed.
        """)
        
        # Get unique roles from ground truth
        roles_info = st.session_state.ground_truth_df[['department', 'role', 'rank']].drop_duplicates().sort_values(['department', 'rank'])
        
        # Create interactive table for observation rates
        st.markdown("**Observation Rates by Role:**")
        
        # Group by department for better organization
        for dept in roles_info['department'].unique():
            if dept == 'Executive':
                continue
                
            with st.expander(f"{dept} Department", expanded=True):
                dept_roles = roles_info[roles_info['department'] == dept]
                
                for idx, row in dept_roles.iterrows():
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.write(f"**{row['role']}** (Rank {row['rank']})")
                    
                    with col2:
                        # Default: lower ranks = lower observation
                        default_rate = 0.5 if row['rank'] <= 2 else (0.7 if row['rank'] <= 4 else 1.0)
                        rate = st.number_input(
                            "Rate",
                            min_value=0.0,
                            max_value=1.0,
                            value=default_rate,
                            step=0.05,
                            key=f"obs_rate_{row['role']}",
                            label_visibility="collapsed"
                        )
                    
                    with col3:
                        st.write(f"{rate*100:.0f}% observed")
        
        # CEO always 100%
        st.markdown("**Executive:**")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.write("**CEO**")
        with col2:
            ceo_rate = st.number_input("Rate", 0.0, 1.0, 1.0, 0.05, 
                                       key="obs_rate_CEO", label_visibility="collapsed")
        with col3:
            st.write(f"{ceo_rate*100:.0f}% observed")
        
        if st.button("Apply Observation Bias", type="primary"):
            bias_sim = ObservationBiasSimulator(st.session_state.ground_truth_df)
            
            # Collect all rates from session state
            for role in roles_info['role'].unique():
                if f"obs_rate_{role}" in st.session_state:
                    bias_sim.set_observation_rate(role, st.session_state[f"obs_rate_{role}"])
            bias_sim.set_observation_rate("CEO", st.session_state.get("obs_rate_CEO", 1.0))
            
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
    
    if st.session_state.biased_df is None:
        st.warning("Please apply observation bias in 'Company Builder' first.")
        st.stop()
    
    st.markdown("""
    Run MVR algorithm on both ground truth and biased companies.
    Compare whether the algorithm correctly identifies hierarchy levels despite selection bias.
    """)
    
    # Show current data status
    st.info(f"Loaded: Ground Truth ({len(st.session_state.ground_truth_df)} records, "
            f"{st.session_state.ground_truth_df['worker_id'].nunique()} workers) | "
            f"Biased ({len(st.session_state.biased_df)} records, "
            f"{st.session_state.biased_df['worker_id'].nunique()} workers)")
    
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
                                 ["Average Optimal Ranking Variance (Default)", 
                                  "Bonhomme et al. (2019) - Paper Method", 
                                  "Elbow (Manual)"],
                                 index=0,
                                 help="Bonhomme method matches the paper exactly; Default is more intuitive")
    
    chosen_K = None
    if kmeans_method == "Elbow (Manual)":
        chosen_K = st.number_input("Choose K", 1, 20, 4)
    
    # Import MVR functions from original code
    def build_bipartite_graph(panel_df):
        """Build bipartite graph of workers and jobs"""
        G = nx.Graph()
        
        for worker, group in panel_df.groupby('worker_id'):
            path = group.sort_values('year')['role'].tolist()
            
            # Remove consecutive duplicates
            path_unique = [path[0]]
            for role in path[1:]:
                if role != path_unique[-1]:
                    path_unique.append(role)
            
            # Add worker node (bipartite=0) and job nodes (bipartite=1)
            G.add_node(worker, bipartite=0)
            for role in path_unique:
                if role not in G:
                    G.add_node(role, bipartite=1)
                G.add_edge(worker, role)
        
        return G
    
    def prune_ilm_network(G, X=10):
        """
        Paper's leave-X-percent-out pruning procedure (Algorithm 1).
        Removes articulation points (cut vertices) that when removed affect less than X% of jobs.
        
        Args:
            G: Bipartite graph with worker (bipartite=0) and job (bipartite=1) nodes
            X: Threshold percentage (default 10%)
        
        Returns:
            Pruned graph G after removing problematic workers
        """
        # Get all job nodes
        job_nodes = {n for n, d in G.nodes(data=True) if d.get('bipartite') == 1}
        
        # Compute degree for each job
        job_degrees = {job: G.degree(job) for job in job_nodes}
        
        # Create G' where each link entering job v is duplicated 100/(d_v * X) times
        # This effectively weights the importance of each link
        G_prime = nx.Graph()
        for u, v in G.edges():
            G_prime.add_edge(u, v)
            # If v is a job, duplicate this edge based on its degree
            if v in job_nodes and job_degrees[v] > 0:
                duplication_factor = int(np.ceil(100 / (job_degrees[v] * X)))
                for _ in range(duplication_factor - 1):  # -1 because already added once
                    G_prime.add_edge(u, v)
        
        # Copy node attributes
        for node, data in G.nodes(data=True):
            if node not in G_prime:
                G_prime.add_node(node, **data)
            else:
                for key, value in data.items():
                    G_prime.nodes[node][key] = value
        
        # Find articulation points in G'
        articulation_points = set(nx.articulation_points(G_prime))
        
        # Filter to only worker articulation points
        worker_nodes = {n for n, d in G.nodes(data=True) if d.get('bipartite') == 0}
        workers_to_remove = articulation_points & worker_nodes
        
        # Remove these workers from original graph G
        G_pruned = G.copy()
        G_pruned.remove_nodes_from(workers_to_remove)
        
        return G_pruned
    
    def get_largest_connected_component(G):
        """Get largest connected component from bipartite graph"""
        if len(G.nodes()) == 0:
            return nx.Graph()
        
        # Find all connected components
        components = list(nx.connected_components(G))
        
        # Return subgraph of largest component
        if components:
            largest_component = max(components, key=len)
            return G.subgraph(largest_component).copy()
        return nx.Graph()
    
    def build_directed_graph_all_pairs(panel_df):
        """Build directed graph H using ALL PAIRS method"""
        H = nx.DiGraph()
        
        for worker, group in panel_df.groupby('worker_id'):
            path = group.sort_values('year')['role'].tolist()
            
            # Remove consecutive duplicates (worker stays in same role)
            path_unique = [path[0]]
            for role in path[1:]:
                if role != path_unique[-1]:
                    path_unique.append(role)
            
            # Create ALL PAIRS edges
            for i in range(len(path_unique)):
                for j in range(i + 1, len(path_unique)):
                    if H.has_edge(path_unique[i], path_unique[j]):
                        H[path_unique[i]][path_unique[j]]['weight'] += 1
                    else:
                        H.add_edge(path_unique[i], path_unique[j], weight=1)
        
        return H
    
    def compute_violations_fast(H, ranking):
        """
        Count violations (paper method: unweighted).
        A violation is an edge (u,v) where rank(u) > rank(v).
        """
        rank_dict = {job: i for i, job in enumerate(ranking)}
        violations = sum(
            1  # Count edges, not weights (paper method)
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
        """Bonhomme method with factorized layer numbering"""
        positions = defaultdict(list)
        for ranking in optimal_rankings:
            for pos, job in enumerate(ranking):
                positions[job].append(pos)
        
        jobs = sorted(positions.keys(), key=lambda x: np.mean(positions[x]))
        N = len(jobs)
        sum_var = sum(np.var(positions[j], ddof=1) if len(positions[j]) > 1 else 0 for j in jobs)
        threshold = (1 / (N - 1)) * sum_var * N
        
        Q_values = []
        for K in range(1, N + 1):
            if K == N:
                Q_K = 0.0
            else:
                kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
                job_mean_ranks = np.array([np.mean(positions[j]) for j in jobs]).reshape(-1, 1)
                kmeans.fit(job_mean_ranks)
                Q_K = kmeans.inertia_
            
            Q_values.append(Q_K)
            if Q_K <= threshold:
                # Factorize labels to get consecutive layer numbers (0, 1, 2, ...)
                kmeans_final = KMeans(n_clusters=K, random_state=0, n_init=10)
                job_mean_ranks = np.array([np.mean(positions[j]) for j in jobs]).reshape(-1, 1)
                labels = kmeans_final.fit_predict(job_mean_ranks)
                
                # Sort jobs by average rank and factorize labels
                labels_sorted = [labels[jobs.index(j)] for j in jobs]
                layer_ids = pd.factorize(labels_sorted)[0]
                
                # Map back to jobs
                labels_factorized = {jobs[i]: layer_ids[i] for i in range(len(jobs))}
                
                return {'K': K, 'threshold': threshold, 'Q_values': Q_values, 
                       'positions': positions, 'labels': labels_factorized}
        
        return {'K': N, 'threshold': threshold, 'Q_values': Q_values, 
               'positions': positions, 'labels': {j: i for i, j in enumerate(jobs)}}
    
    def run_kmeans_overall_std(optimal_rankings):
        """Overall Std (Average Optimal Ranking Variance) method with factorized layer numbering"""
        positions = defaultdict(list)
        for ranking in optimal_rankings:
            for pos, job in enumerate(ranking):
                positions[job].append(pos)
        
        jobs = sorted(positions.keys(), key=lambda x: np.mean(positions[x]))
        N = len(jobs)
        
        average_ranks = {job: np.mean(positions[job]) for job in jobs}
        rank_var = np.var(list(average_ranks.values()))
        threshold = (1 / (N - 1)) * rank_var * N
        
        Q_values = []
        for K in range(1, N + 1):
            if K == N:
                Q_K = 0.0
            else:
                kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
                job_mean_ranks = np.array([np.mean(positions[j]) for j in jobs]).reshape(-1, 1)
                kmeans.fit(job_mean_ranks)
                Q_K = kmeans.inertia_
            
            Q_values.append(Q_K)
            if Q_K <= threshold:
                # Factorize labels to get consecutive layer numbers
                kmeans_final = KMeans(n_clusters=K, random_state=0, n_init=10)
                job_mean_ranks = np.array([np.mean(positions[j]) for j in jobs]).reshape(-1, 1)
                labels = kmeans_final.fit_predict(job_mean_ranks)
                
                # Sort jobs by average rank and factorize labels
                labels_sorted = [labels[jobs.index(j)] for j in jobs]
                layer_ids = pd.factorize(labels_sorted)[0]
                
                # Map back to jobs
                labels_factorized = {jobs[i]: layer_ids[i] for i in range(len(jobs))}
                
                return {'K': K, 'threshold': threshold, 'Q_values': Q_values, 
                       'positions': positions, 'labels': labels_factorized}
        
        return {'K': N, 'threshold': threshold, 'Q_values': Q_values, 
               'positions': positions, 'labels': {j: i for i, j in enumerate(jobs)}}
    
    # Run analysis
    # ILM Pruning Settings
    st.subheader("3. ILM Network Pruning (Optional)")
    
    col1, col2 = st.columns(2)
    with col1:
        enable_pruning = st.checkbox("Enable ILM Pruning (Paper's Algorithm 1)", value=True,
                                     help="Remove articulation points to address measurement error")
    with col2:
        X_threshold = st.slider("X% Threshold", min_value=5, max_value=20, value=10, step=5,
                               help="Remove workers affecting less than X% of jobs")
    
    if st.button("Run MVR Analysis on Both Companies", type="primary"):
        
        # Run both companies with ILM pruning if enabled
        with st.spinner("Running MVR on Ground Truth Company..."):
            if enable_pruning:
                # Step 1: Build bipartite graph
                G_gt = build_bipartite_graph(st.session_state.ground_truth_df)
                st.info(f"Ground Truth - Initial bipartite graph: {len([n for n, d in G_gt.nodes(data=True) if d.get('bipartite')==1])} jobs, {len([n for n, d in G_gt.nodes(data=True) if d.get('bipartite')==0])} workers")
                
                # Step 2: Prune network
                G_gt_pruned = prune_ilm_network(G_gt, X=X_threshold)
                workers_removed = len([n for n, d in G_gt.nodes(data=True) if d.get('bipartite')==0]) - len([n for n, d in G_gt_pruned.nodes(data=True) if d.get('bipartite')==0])
                st.info(f"Ground Truth - Removed {workers_removed} articulation point workers")
                
                # Step 3: Get largest connected component (largest ILM)
                G_gt_ilm = get_largest_connected_component(G_gt_pruned)
                jobs_in_ilm = len([n for n, d in G_gt_ilm.nodes(data=True) if d.get('bipartite')==1])
                st.info(f"Ground Truth - Largest ILM contains {jobs_in_ilm} jobs")
                
                # Step 4: Build directed graph from largest ILM
                # Filter panel_df to only include workers and jobs in largest ILM
                ilm_workers = {n for n, d in G_gt_ilm.nodes(data=True) if d.get('bipartite')==0}
                ilm_jobs = {n for n, d in G_gt_ilm.nodes(data=True) if d.get('bipartite')==1}
                panel_gt_filtered = st.session_state.ground_truth_df[
                    (st.session_state.ground_truth_df['worker_id'].isin(ilm_workers)) &
                    (st.session_state.ground_truth_df['role'].isin(ilm_jobs))
                ]
                H_gt = build_directed_graph_all_pairs(panel_gt_filtered)
            else:
                H_gt = build_directed_graph_all_pairs(st.session_state.ground_truth_df)
            
            opt_gt, viol_gt, prog_gt = find_optimal_rankings_mvr(H_gt, R, T, early_stop)
        
        with st.spinner("Running MVR on Biased Company..."):
            if enable_pruning:
                # Same steps for biased company
                G_bias = build_bipartite_graph(st.session_state.biased_df)
                st.info(f"Biased - Initial bipartite graph: {len([n for n, d in G_bias.nodes(data=True) if d.get('bipartite')==1])} jobs, {len([n for n, d in G_bias.nodes(data=True) if d.get('bipartite')==0])} workers")
                
                G_bias_pruned = prune_ilm_network(G_bias, X=X_threshold)
                workers_removed = len([n for n, d in G_bias.nodes(data=True) if d.get('bipartite')==0]) - len([n for n, d in G_bias_pruned.nodes(data=True) if d.get('bipartite')==0])
                st.info(f"Biased - Removed {workers_removed} articulation point workers")
                
                G_bias_ilm = get_largest_connected_component(G_bias_pruned)
                jobs_in_ilm = len([n for n, d in G_bias_ilm.nodes(data=True) if d.get('bipartite')==1])
                st.info(f"Biased - Largest ILM contains {jobs_in_ilm} jobs")
                
                ilm_workers = {n for n, d in G_bias_ilm.nodes(data=True) if d.get('bipartite')==0}
                ilm_jobs = {n for n, d in G_bias_ilm.nodes(data=True) if d.get('bipartite')==1}
                panel_bias_filtered = st.session_state.biased_df[
                    (st.session_state.biased_df['worker_id'].isin(ilm_workers)) &
                    (st.session_state.biased_df['role'].isin(ilm_jobs))
                ]
                H_bias = build_directed_graph_all_pairs(panel_bias_filtered)
            else:
                H_bias = build_directed_graph_all_pairs(st.session_state.biased_df)
            
            opt_bias, viol_bias, prog_bias = find_optimal_rankings_mvr(H_bias, R, T, early_stop)
        
        # Run K-means
        if kmeans_method.startswith("Average Optimal Ranking Variance"):
            result_gt = run_kmeans_overall_std(opt_gt)
            result_bias = run_kmeans_overall_std(opt_bias)
        elif kmeans_method.startswith("Bonhomme"):
            result_gt = run_kmeans_bonhomme(opt_gt)
            result_bias = run_kmeans_bonhomme(opt_bias)
        else:  # Elbow (Manual)
            # For manual K selection, we still need to get positions but override K
            result_gt = run_kmeans_overall_std(opt_gt)
            result_bias = run_kmeans_overall_std(opt_bias)
            
            # Override with user's chosen K and recompute labels
            if chosen_K is not None:
                # Recompute K-means with chosen K for ground truth
                positions_gt = result_gt['positions']
                jobs_gt = sorted(positions_gt.keys(), key=lambda x: np.mean(positions_gt[x]))
                kmeans_gt = KMeans(n_clusters=chosen_K, random_state=0, n_init=10)
                job_mean_ranks_gt = np.array([np.mean(positions_gt[j]) for j in jobs_gt]).reshape(-1, 1)
                labels_gt = kmeans_gt.fit_predict(job_mean_ranks_gt)
                labels_sorted_gt = [labels_gt[jobs_gt.index(j)] for j in jobs_gt]
                layer_ids_gt = pd.factorize(labels_sorted_gt)[0]
                result_gt['K'] = chosen_K
                result_gt['labels'] = {jobs_gt[i]: layer_ids_gt[i] for i in range(len(jobs_gt))}
                
                # Same for biased
                positions_bias = result_bias['positions']
                jobs_bias = sorted(positions_bias.keys(), key=lambda x: np.mean(positions_bias[x]))
                kmeans_bias = KMeans(n_clusters=chosen_K, random_state=0, n_init=10)
                job_mean_ranks_bias = np.array([np.mean(positions_bias[j]) for j in jobs_bias]).reshape(-1, 1)
                labels_bias = kmeans_bias.fit_predict(job_mean_ranks_bias)
                labels_sorted_bias = [labels_bias[jobs_bias.index(j)] for j in jobs_bias]
                layer_ids_bias = pd.factorize(labels_sorted_bias)[0]
                result_bias['K'] = chosen_K
                result_bias['labels'] = {jobs_bias[i]: layer_ids_bias[i] for i in range(len(jobs_bias))}
        
        K_gt = result_gt['K']
        K_bias = result_bias['K']
        
        # Store results in session state for detailed view
        st.session_state.mvr_results = {
            'H_gt': H_gt, 'H_bias': H_bias,
            'opt_gt': opt_gt, 'opt_bias': opt_bias,
            'viol_gt': viol_gt, 'viol_bias': viol_bias,
            'prog_gt': prog_gt, 'prog_bias': prog_bias,
            'result_gt': result_gt, 'result_bias': result_bias,
            'K_gt': K_gt, 'K_bias': K_bias
        }
        
        # Display summary comparison
        st.subheader("Summary Comparison")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ground Truth K", K_gt)
        with col2:
            st.metric("Biased Data K", K_bias)
        with col3:
            diff = abs(K_gt - K_bias)
            st.metric("Difference", diff)
        
        if K_gt == K_bias:
            st.success("Algorithm correctly identified the same number of layers despite selection bias.")
        else:
            st.warning(f"Algorithm identified different layer counts: GT={K_gt}, Biased={K_bias}")
        
        # Job Cluster Visualization
        st.markdown("**Job Cluster Visualization**")
        
        # Get positions and labels from K-means results (already factorized)
        positions_gt = result_gt['positions']
        positions_bias = result_bias['positions']
        labels_gt_dict = result_gt.get('labels', {})
        labels_bias_dict = result_bias.get('labels', {})
        
        jobs_gt = sorted(positions_gt.keys(), key=lambda x: np.mean(positions_gt[x]))
        jobs_bias = sorted(positions_bias.keys(), key=lambda x: np.mean(positions_bias[x]))
        
        # Create side-by-side cluster plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        
        # Color palette for clusters
        colors = plt.cm.Set3(np.linspace(0, 1, max(K_gt, K_bias)))
        
        # Ground Truth: Sort by average rank (lower rank = higher position)
        job_ranks_gt = [(job, np.mean(positions_gt[job]), labels_gt_dict.get(job, 0)) 
                        for job in jobs_gt]
        job_ranks_gt.sort(key=lambda x: x[1])  # Sort by average rank ascending
        
        jobs_sorted_gt = [j[0] for j in job_ranks_gt]
        ranks_sorted_gt = [j[1] for j in job_ranks_gt]
        colors_gt = [colors[j[2] % len(colors)] for j in job_ranks_gt]
        
        y_pos_gt = range(len(jobs_sorted_gt))
        ax1.barh(y_pos_gt, ranks_sorted_gt, color=colors_gt, alpha=0.8,
                edgecolor='black', linewidth=1.5, height=0.8)
        ax1.set_yticks(y_pos_gt)
        ax1.set_yticklabels(jobs_sorted_gt, fontsize=11, fontweight='bold')
        ax1.set_xlabel('Average Position in Ranking', fontsize=12, fontweight='bold')
        ax1.set_title(f'Ground Truth Company (K={K_gt})', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()  # Higher ranks (lower values) at top
        
        # Biased Company: Sort by average rank
        job_ranks_bias = [(job, np.mean(positions_bias[job]), labels_bias_dict.get(job, 0)) 
                          for job in jobs_bias]
        job_ranks_bias.sort(key=lambda x: x[1])
        
        jobs_sorted_bias = [j[0] for j in job_ranks_bias]
        ranks_sorted_bias = [j[1] for j in job_ranks_bias]
        colors_bias = [colors[j[2] % len(colors)] for j in job_ranks_bias]
        
        y_pos_bias = range(len(jobs_sorted_bias))
        ax2.barh(y_pos_bias, ranks_sorted_bias, color=colors_bias, alpha=0.8,
                edgecolor='black', linewidth=1.5, height=0.8)
        ax2.set_yticks(y_pos_bias)
        ax2.set_yticklabels(jobs_sorted_bias, fontsize=11, fontweight='bold')
        ax2.set_xlabel('Average Position in Ranking', fontsize=12, fontweight='bold')
        ax2.set_title(f'Biased Company (K={K_bias})', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Ranking Comparison Visualization
        st.markdown("**Job Ranking Comparison**")
        
        # Get positions for both companies
        positions_gt = result_gt['positions']
        positions_bias = result_bias['positions']
        
        # Calculate average ranks
        jobs_gt = sorted(positions_gt.keys(), key=lambda x: np.mean(positions_gt[x]))
        jobs_bias = sorted(positions_bias.keys(), key=lambda x: np.mean(positions_bias[x]))
        
        # Create side-by-side ranking plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Ground Truth ranking
        means_gt = [np.mean(positions_gt[j]) for j in jobs_gt]
        stds_gt = [np.std(positions_gt[j], ddof=1) if len(positions_gt[j]) > 1 else 0 for j in jobs_gt]
        
        ax1.barh(range(len(jobs_gt)), means_gt, xerr=stds_gt, capsize=5, alpha=0.7, color='steelblue')
        ax1.set_yticks(range(len(jobs_gt)))
        ax1.set_yticklabels(jobs_gt)
        ax1.set_xlabel('Average Rank (± Std Dev)', fontsize=11)
        ax1.set_title('Ground Truth Company Rankings', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Biased company ranking
        means_bias = [np.mean(positions_bias[j]) for j in jobs_bias]
        stds_bias = [np.std(positions_bias[j], ddof=1) if len(positions_bias[j]) > 1 else 0 for j in jobs_bias]
        
        ax2.barh(range(len(jobs_bias)), means_bias, xerr=stds_bias, capsize=5, alpha=0.7, color='coral')
        ax2.set_yticks(range(len(jobs_bias)))
        ax2.set_yticklabels(jobs_bias)
        ax2.set_xlabel('Average Rank (± Std Dev)', fontsize=11)
        ax2.set_title('Biased Company Rankings', fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Convergence Comparison
        st.subheader("Step 1: MVR Convergence Comparison")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(range(1, len(prog_gt) + 1), prog_gt, 'b-', linewidth=2)
        ax1.set_title('Ground Truth Company', fontweight='bold')
        ax1.set_xlabel('Number of Repetitions')
        ax1.set_ylabel('Unique Optimal Rankings Found')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(range(1, len(prog_bias) + 1), prog_bias, 'r-', linewidth=2)
        ax2.set_title('Biased Company', fontweight='bold')
        ax2.set_xlabel('Number of Repetitions')
        ax2.set_ylabel('Unique Optimal Rankings Found')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Graph: {H_gt.number_of_nodes()} nodes, {H_gt.number_of_edges()} edges | "
                   f"Found {len(opt_gt)} optimal rankings | Min violations: {viol_gt}")
        with col2:
            st.info(f"Graph: {H_bias.number_of_nodes()} nodes, {H_bias.number_of_edges()} edges | "
                   f"Found {len(opt_bias)} optimal rankings | Min violations: {viol_bias}")
        
        # Detailed Results - Ground Truth
        st.markdown("---")
        st.subheader("Detailed Analysis: Ground Truth Company")
        
        positions_gt = result_gt['positions']
        stats_gt = {}
        for job, pos_list in positions_gt.items():
            stats_gt[job] = {
                'mean': np.mean(pos_list),
                'std': np.std(pos_list, ddof=1) if len(pos_list) > 1 else 0,
                'var': np.var(pos_list, ddof=1) if len(pos_list) > 1 else 0,
                'min': float(min(pos_list)),
                'max': float(max(pos_list)),
                'unique': len(set(pos_list))
            }
        stats_df_gt = pd.DataFrame(stats_gt).T.sort_values('mean')
        jobs_sorted_gt = stats_df_gt.index.tolist()
        
        st.markdown("**Step 2: Job Position Variance**")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            means = [stats_gt[j]['mean'] for j in jobs_sorted_gt]
            stds = [stats_gt[j]['std'] for j in jobs_sorted_gt]
            ax.barh(range(len(jobs_sorted_gt)), means, xerr=stds, capsize=5, alpha=0.7, color='steelblue')
            ax.set_yticks(range(len(jobs_sorted_gt)))
            ax.set_yticklabels(jobs_sorted_gt)
            ax.set_xlabel('Average Rank (± Std Dev)')
            ax.set_title('Average Ranking with Uncertainty', fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            variances = [stats_gt[j]['var'] for j in jobs_sorted_gt]
            colors = ['salmon' if v > np.median(variances) else 'lightgreen' for v in variances]
            ax.barh(range(len(jobs_sorted_gt)), variances, alpha=0.7, color=colors)
            ax.set_yticks(range(len(jobs_sorted_gt)))
            ax.set_yticklabels(jobs_sorted_gt)
            ax.set_xlabel('Position Variance')
            ax.set_title('Job Position Variance', fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("**Job Position Statistics**")
        display_df_gt = stats_df_gt.copy()
        display_df_gt.index.name = 'Job'
        display_df_gt = display_df_gt.reset_index()
        display_df_gt['mean'] = display_df_gt['mean'].apply(lambda x: f"{x:.2f}")
        display_df_gt['std'] = display_df_gt['std'].apply(lambda x: f"{x:.2f}")
        display_df_gt['var'] = display_df_gt['var'].apply(lambda x: f"{x:.2f}")
        display_df_gt['min'] = display_df_gt['min'].apply(lambda x: f"{x:.1f}")
        display_df_gt['max'] = display_df_gt['max'].apply(lambda x: f"{x:.1f}")
        display_df_gt['unique'] = display_df_gt['unique'].apply(lambda x: f"{x:.1f}")
        st.dataframe(display_df_gt, use_container_width=True, hide_index=True)
        
        st.markdown("**Step 3: K-means Clustering**")
        st.info(f"Method: {kmeans_method} | Optimal K: {K_gt} | Threshold: {result_gt['threshold']:.4f}")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        K_range = range(1, len(result_gt['Q_values']) + 1)
        ax.plot(K_range, result_gt['Q_values'], 'bo-', linewidth=2)
        ax.axhline(y=result_gt['threshold'], color='r', linestyle='--', linewidth=2,
                  label=f"Threshold: {result_gt['threshold']:.2f}")
        ax.axvline(x=K_gt, color='g', linestyle=':', linewidth=2, label=f"Optimal K={K_gt}")
        ax.set_xlabel('K')
        ax.set_ylabel('Q(K)')
        ax.set_title(f'{kmeans_method}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Detailed Results - Biased Company
        st.markdown("---")
        st.subheader("Detailed Analysis: Biased Company")
        
        positions_bias = result_bias['positions']
        stats_bias = {}
        for job, pos_list in positions_bias.items():
            stats_bias[job] = {
                'mean': np.mean(pos_list),
                'std': np.std(pos_list, ddof=1) if len(pos_list) > 1 else 0,
                'var': np.var(pos_list, ddof=1) if len(pos_list) > 1 else 0,
                'min': float(min(pos_list)),
                'max': float(max(pos_list)),
                'unique': len(set(pos_list))
            }
        stats_df_bias = pd.DataFrame(stats_bias).T.sort_values('mean')
        jobs_sorted_bias = stats_df_bias.index.tolist()
        
        st.markdown("**Step 2: Job Position Variance**")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            means = [stats_bias[j]['mean'] for j in jobs_sorted_bias]
            stds = [stats_bias[j]['std'] for j in jobs_sorted_bias]
            ax.barh(range(len(jobs_sorted_bias)), means, xerr=stds, capsize=5, alpha=0.7, color='steelblue')
            ax.set_yticks(range(len(jobs_sorted_bias)))
            ax.set_yticklabels(jobs_sorted_bias)
            ax.set_xlabel('Average Rank (± Std Dev)')
            ax.set_title('Average Ranking with Uncertainty', fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            variances = [stats_bias[j]['var'] for j in jobs_sorted_bias]
            colors = ['salmon' if v > np.median(variances) else 'lightgreen' for v in variances]
            ax.barh(range(len(jobs_sorted_bias)), variances, alpha=0.7, color=colors)
            ax.set_yticks(range(len(jobs_sorted_bias)))
            ax.set_yticklabels(jobs_sorted_bias)
            ax.set_xlabel('Position Variance')
            ax.set_title('Job Position Variance', fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("**Job Position Statistics**")
        display_df_bias = stats_df_bias.copy()
        display_df_bias.index.name = 'Job'
        display_df_bias = display_df_bias.reset_index()
        display_df_bias['mean'] = display_df_bias['mean'].apply(lambda x: f"{x:.2f}")
        display_df_bias['std'] = display_df_bias['std'].apply(lambda x: f"{x:.2f}")
        display_df_bias['var'] = display_df_bias['var'].apply(lambda x: f"{x:.2f}")
        display_df_bias['min'] = display_df_bias['min'].apply(lambda x: f"{x:.1f}")
        display_df_bias['max'] = display_df_bias['max'].apply(lambda x: f"{x:.1f}")
        display_df_bias['unique'] = display_df_bias['unique'].apply(lambda x: f"{x:.1f}")
        st.dataframe(display_df_bias, use_container_width=True, hide_index=True)
        
        st.markdown("**Step 3: K-means Clustering**")
        st.info(f"Method: {kmeans_method} | Optimal K: {K_bias} | Threshold: {result_bias['threshold']:.4f}")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        K_range = range(1, len(result_bias['Q_values']) + 1)
        ax.plot(K_range, result_bias['Q_values'], 'bo-', linewidth=2)
        ax.axhline(y=result_bias['threshold'], color='r', linestyle='--', linewidth=2,
                  label=f"Threshold: {result_bias['threshold']:.2f}")
        ax.axvline(x=K_bias, color='g', linestyle=':', linewidth=2, label=f"Optimal K={K_bias}")
        ax.set_xlabel('K')
        ax.set_ylabel('Q(K)')
        ax.set_title(f'{kmeans_method}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

if __name__ == "__main__":
    pass
