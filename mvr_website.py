"""
MVR Website: Interactive Hierarchy Inference Tool

This Streamlit app demonstrates the robustness of the MVR algorithm
for inferring organizational hierarchies from LinkedIn career transition data.
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

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="MVR Hierarchy Inference",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== HELPER FUNCTIONS ====================

@st.cache_data
def create_test_data():
    """Create deterministic test data"""
    data = []
    worker_id = 1
    
    # Sales workers
    sales_path = ['Sales1', 'Sales2', 'Sales3', 'Sales4', 'Sales5', 'Sales6', 'CEO']
    for _ in range(10):
        for year, role in enumerate(sales_path, start=2010):
            data.append({'Worker': worker_id, 'Year': year, 'Role': role})
        worker_id += 1
    
    # Engineering workers
    eng_path = ['Eng1', 'Eng2', 'Eng3', 'Eng4', 'CEO']
    for _ in range(10):
        for year, role in enumerate(eng_path, start=2010):
            data.append({'Worker': worker_id, 'Year': year, 'Role': role})
        worker_id += 1
    
    return pd.DataFrame(data)

def build_directed_graph_all_pairs(panel_df):
    """Build directed graph H using ALL PAIRS method"""
    H = nx.DiGraph()
    
    for worker, group in panel_df.groupby('Worker'):
        path = group.sort_values('Year')['Role'].tolist()
        
        # Create ALL PAIRS edges
        for i in range(len(path)):
            for j in range(i + 1, len(path)):
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

def find_optimal_rankings_mvr(H, R=3000, T=1500, convergence_window=500, early_stop=False, seed=42):
    """MVR: pure random swap with S' <= S acceptance"""
    random.seed(seed)
    np.random.seed(seed)
    
    jobs = list(H.nodes())
    n = len(jobs)
    
    # Initial ranking by out-degree
    jobs.sort(key=lambda x: H.out_degree(x, weight='weight'), reverse=True)
    current_ranking = jobs.copy()
    current_violations = compute_violations_fast(H, current_ranking)
    
    min_violations = current_violations
    optimal_rankings = set()
    optimal_rankings.add(tuple(current_ranking))
    
    progress = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for r in range(R):
        for t in range(T):
            # Pure random swap
            i, j = random.sample(range(n), 2)
            new_ranking = current_ranking.copy()
            new_ranking[i], new_ranking[j] = new_ranking[j], new_ranking[i]
            new_violations = compute_violations_fast(H, new_ranking)
            
            # Accept if S' <= S
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
        
        # Update progress
        progress_bar.progress((r + 1) / R)
        if (r + 1) % 100 == 0:
            status_text.text(f"Repetition {r + 1}/{R}: {len(optimal_rankings)} unique rankings found")
        
        # Early stopping
        if early_stop and r >= convergence_window:
            recent = progress[-convergence_window:]
            if len(set(recent)) == 1:
                status_text.text(f"Early stopping at R={r+1} (converged)")
                progress_bar.progress(1.0)
                break
    
    progress_bar.empty()
    status_text.empty()
    
    return list(optimal_rankings), min_violations, progress

# K-means methods
def method_bonhomme(optimal_rankings):
    """Bonhomme et al. (2019) - Rank Std Threshold"""
    positions = defaultdict(list)
    for ranking in optimal_rankings:
        for pos, job in enumerate(ranking):
            positions[job].append(pos)
    
    jobs = sorted(positions.keys(), key=lambda x: np.mean(positions[x]))
    job_mean_ranks = np.array([np.mean(positions[j]) for j in jobs]).reshape(-1, 1)
    N = len(jobs)
    
    sum_var = sum(np.var(positions[j], ddof=1) if len(positions[j]) > 1 else 0 for j in jobs)
    threshold = (1 / (N - 1)) * sum_var * N
    
    optimal_K = 1
    Q_values = []
    
    for K in range(1, N + 1):
        if K == N:
            Q_K = 0.0
        else:
            kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
            kmeans.fit(job_mean_ranks)
            Q_K = kmeans.inertia_
        
        Q_values.append(Q_K)
        if Q_K <= threshold:
            optimal_K = K
            break
    
    if optimal_K == N:
        labels = np.arange(N)
    else:
        kmeans = KMeans(n_clusters=optimal_K, random_state=0, n_init=10)
        labels = kmeans.fit_predict(job_mean_ranks)
    
    return {
        'K': optimal_K, 'threshold': threshold, 'Q_values': Q_values,
        'labels': labels, 'jobs': jobs, 'mean_ranks': job_mean_ranks
    }

def method_overall_std(optimal_rankings):
    """Overall Std Threshold"""
    positions = defaultdict(list)
    for ranking in optimal_rankings:
        for pos, job in enumerate(ranking):
            positions[job].append(pos)
    
    jobs = sorted(positions.keys(), key=lambda x: np.mean(positions[x]))
    job_mean_ranks = np.array([np.mean(positions[j]) for j in jobs]).reshape(-1, 1)
    N = len(jobs)
    
    average_ranks = {job: np.mean(positions[job]) for job in jobs}
    rank_var = np.var(list(average_ranks.values()))
    threshold = (1 / (N - 1)) * rank_var * N
    
    optimal_K = 1
    Q_values = []
    
    for K in range(1, N + 1):
        if K == N:
            Q_K = 0.0
        else:
            kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
            kmeans.fit(job_mean_ranks)
            Q_K = kmeans.inertia_
        
        Q_values.append(Q_K)
        if Q_K <= threshold:
            optimal_K = K
            break
    
    if optimal_K == N:
        labels = np.arange(N)
    else:
        kmeans = KMeans(n_clusters=optimal_K, random_state=0, n_init=10)
        labels = kmeans.fit_predict(job_mean_ranks)
    
    return {
        'K': optimal_K, 'threshold': threshold, 'Q_values': Q_values,
        'labels': labels, 'jobs': jobs, 'mean_ranks': job_mean_ranks
    }

def method_elbow(optimal_rankings):
    """Elbow method - returns data for manual K selection"""
    positions = defaultdict(list)
    for ranking in optimal_rankings:
        for pos, job in enumerate(ranking):
            positions[job].append(pos)
    
    jobs = sorted(positions.keys(), key=lambda x: np.mean(positions[x]))
    job_mean_ranks = np.array([np.mean(positions[j]) for j in jobs]).reshape(-1, 1)
    N = len(jobs)
    
    inertias = []
    K_range = range(1, N + 1)
    for K in K_range:
        if K == 1:
            inertias.append(np.var(job_mean_ranks) * N)
        elif K == N:
            inertias.append(0.0)
        else:
            kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
            kmeans.fit(job_mean_ranks)
            inertias.append(kmeans.inertia_)
    
    return {
        'inertias': inertias, 'jobs': jobs, 'mean_ranks': job_mean_ranks,
        'K_range': list(K_range)
    }

def apply_elbow_K(elbow_data, chosen_K):
    """Apply manually chosen K"""
    jobs = elbow_data['jobs']
    mean_ranks = elbow_data['mean_ranks']
    N = len(jobs)
    
    if chosen_K == N:
        labels = np.arange(N)
    else:
        kmeans = KMeans(n_clusters=chosen_K, random_state=0, n_init=10)
        labels = kmeans.fit_predict(mean_ranks)
    
    return {
        'K': chosen_K, 'inertias': elbow_data['inertias'],
        'labels': labels, 'jobs': jobs, 'mean_ranks': mean_ranks,
        'K_range': elbow_data['K_range']
    }

def method_simple_var(optimal_rankings):
    """Simple Variance Threshold"""
    positions = defaultdict(list)
    for ranking in optimal_rankings:
        for pos, job in enumerate(ranking):
            positions[job].append(pos)
    
    jobs = sorted(positions.keys(), key=lambda x: np.mean(positions[x]))
    job_mean_ranks = np.array([np.mean(positions[j]) for j in jobs]).reshape(-1, 1)
    N = len(jobs)
    
    job_variances = [np.var(positions[j], ddof=1) if len(positions[j]) > 1 else 0 for j in jobs]
    threshold = np.median(job_variances)
    
    optimal_K = 1
    Q_values = []
    
    for K in range(1, N + 1):
        if K == N:
            Q_K = 0.0
        else:
            kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
            kmeans.fit(job_mean_ranks)
            Q_K = kmeans.inertia_ / N
        
        Q_values.append(Q_K)
        if Q_K <= threshold:
            optimal_K = K
            break
    
    if optimal_K == N:
        labels = np.arange(N)
    else:
        kmeans = KMeans(n_clusters=optimal_K, random_state=0, n_init=10)
        labels = kmeans.fit_predict(job_mean_ranks)
    
    return {
        'K': optimal_K, 'threshold': threshold, 'Q_values': Q_values,
        'labels': labels, 'jobs': jobs, 'mean_ranks': job_mean_ranks
    }

# ==================== MAIN APP ====================

def main():
    st.title("MVR Hierarchy Inference Tool")
    st.markdown("""
    This tool demonstrates the **Minimum Violation Ranking (MVR)** algorithm for inferring 
    organizational hierarchies from career transition data.
    
    **Test Case**: 10 Sales workers (6 levels) + 10 Engineering workers (4 levels) to CEO
    """)
    
    # Sidebar
    st.sidebar.header("Algorithm Parameters")
    
    st.sidebar.subheader("MVR Parameters")
    R = st.sidebar.slider("R (Repetitions)", 100, 5000, 3000, 100, 
                          help="Number of independent MVR runs")
    T = st.sidebar.slider("T (Iterations per run)", 100, 3000, 1500, 100,
                          help="Number of swaps per repetition")
    early_stop = st.sidebar.checkbox("Enable Early Stopping", False,
                                     help="Stop if no new rankings found in 500 rounds")
    
    st.sidebar.subheader("K-means Method")
    kmeans_method = st.sidebar.selectbox(
        "Select Method",
        ["Bonhomme et al. (2019)", "Overall Std", "Elbow (Manual)", "Simple Variance"]
    )
    
    if kmeans_method == "Elbow (Manual)":
        chosen_K = st.sidebar.number_input("Choose K (after viewing plot)", 
                                           min_value=1, max_value=11, value=4)
    
    run_button = st.sidebar.button("Run Analysis", type="primary")
    
    # Main content
    if run_button:
        # Step 1: Create data
        with st.spinner("Creating test data..."):
            panel_df = create_test_data()
            H = build_directed_graph_all_pairs(panel_df)
        
        st.success(f"Data created: {panel_df['Worker'].nunique()} workers, {H.number_of_nodes()} roles, {H.number_of_edges()} edges")
        
        # Step 2: Run MVR
        st.subheader("Step 1: MVR - Finding Optimal Rankings")
        with st.spinner(f"Running MVR (R={R}, T={T})..."):
            optimal_rankings, min_violations, progress = find_optimal_rankings_mvr(
                H, R=R, T=T, early_stop=early_stop
            )
        
        st.success(f"Found {len(optimal_rankings)} unique optimal rankings with {min_violations} violations")
        
        # Convergence plot
        col1, col2 = st.columns([2, 1])
        
        with col1:
            n_sales = len([j for j in H.nodes() if 'Sales' in j])
            n_eng = len([j for j in H.nodes() if 'Eng' in j])
            theoretical_max = comb(n_sales + n_eng, n_sales)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(1, len(progress) + 1), progress, 'b-', linewidth=2)
            ax.axhline(y=theoretical_max, color='r', linestyle='--', linewidth=2,
                      label=f'Theoretical max: C({n_sales + n_eng}, {n_sales}) = {theoretical_max}')
            ax.set_xlabel('Number of Repetitions', fontsize=11)
            ax.set_ylabel('Unique Optimal Rankings Found', fontsize=11)
            ax.set_title('Convergence: Discovery of Optimal Rankings Over Time', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            coverage = len(optimal_rankings) / theoretical_max * 100
            st.metric("Coverage", f"{coverage:.1f}%")
            st.metric("Theoretical Max", theoretical_max)
            st.metric("Found", len(optimal_rankings))
        
        # Job variance analysis
        st.subheader("Step 2: Job Position Variance")
        
        positions = defaultdict(list)
        for ranking in optimal_rankings:
            for pos, job in enumerate(ranking):
                positions[job].append(pos)
        
        stats = {}
        for job, pos_list in positions.items():
            stats[job] = {
                'mean': np.mean(pos_list),
                'std': np.std(pos_list, ddof=1) if len(pos_list) > 1 else 0,
                'var': np.var(pos_list, ddof=1) if len(pos_list) > 1 else 0,
            }
        
        stats_df = pd.DataFrame(stats).T
        stats_df = stats_df.sort_values('mean')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            jobs_sorted = stats_df.index.tolist()
            means = [stats[j]['mean'] for j in jobs_sorted]
            stds = [stats[j]['std'] for j in jobs_sorted]
            ax.barh(range(len(jobs_sorted)), means, xerr=stds, capsize=5, alpha=0.7, color='steelblue')
            ax.set_yticks(range(len(jobs_sorted)))
            ax.set_yticklabels(jobs_sorted)
            ax.set_xlabel('Average Rank (± Std Dev)')
            ax.set_title('Average Ranking with Uncertainty', fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            variances = [stats[j]['var'] for j in jobs_sorted]
            colors = ['salmon' if v > np.median(variances) else 'lightgreen' for v in variances]
            ax.barh(range(len(jobs_sorted)), variances, alpha=0.7, color=colors)
            ax.set_yticks(range(len(jobs_sorted)))
            ax.set_yticklabels(jobs_sorted)
            ax.set_xlabel('Position Variance')
            ax.set_title('Job Position Variance', fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        # K-means clustering
        st.subheader("Step 3: K-means Clustering - Hierarchy Levels")
        
        # Run selected method
        if kmeans_method == "Bonhomme et al. (2019)":
            result = method_bonhomme(optimal_rankings)
            st.info(f"**Method**: Bonhomme et al. (2019) - Rank Std Threshold")
            st.write(f"Optimal K: **{result['K']}** | Threshold: **{result['threshold']:.4f}**")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            K_range = range(1, len(result['Q_values']) + 1)
            ax.plot(K_range, result['Q_values'], 'bo-', linewidth=2)
            ax.axhline(y=result['threshold'], color='r', linestyle='--', linewidth=2,
                      label=f"Threshold: {result['threshold']:.2f}")
            ax.axvline(x=result['K'], color='g', linestyle=':', linewidth=2,
                      label=f"Optimal K={result['K']}")
            ax.set_xlabel('K')
            ax.set_ylabel('Q(K)')
            ax.set_title('Bonhomme et al. (2019)', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        elif kmeans_method == "Overall Std":
            result = method_overall_std(optimal_rankings)
            st.info(f"**Method**: Overall Std Threshold")
            st.write(f"Optimal K: **{result['K']}** | Threshold: **{result['threshold']:.4f}**")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            K_range = range(1, len(result['Q_values']) + 1)
            ax.plot(K_range, result['Q_values'], 'bo-', linewidth=2)
            ax.axhline(y=result['threshold'], color='r', linestyle='--', linewidth=2,
                      label=f"Threshold: {result['threshold']:.2f}")
            ax.axvline(x=result['K'], color='g', linestyle=':', linewidth=2,
                      label=f"Optimal K={result['K']}")
            ax.set_xlabel('K')
            ax.set_ylabel('Q(K)')
            ax.set_title('Overall Std Threshold', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        elif kmeans_method == "Elbow (Manual)":
            elbow_data = method_elbow(optimal_rankings)
            result = apply_elbow_K(elbow_data, chosen_K)
            st.info(f"**Method**: Elbow Method (Manual Selection)")
            st.write(f"Chosen K: **{result['K']}**")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(result['K_range'], result['inertias'], 'bo-', linewidth=2, markersize=8)
            ax.axvline(x=result['K'], color='g', linestyle=':', linewidth=2,
                      label=f"Chosen K={result['K']}")
            ax.set_xlabel('Number of Clusters (K)')
            ax.set_ylabel('Inertia')
            ax.set_title('Elbow Method', fontweight='bold')
            ax.set_xticks(result['K_range'])
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        else:  # Simple Variance
            result = method_simple_var(optimal_rankings)
            st.info(f"**Method**: Simple Variance Threshold")
            st.write(f"Optimal K: **{result['K']}** | Threshold: **{result['threshold']:.4f}**")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            K_range = range(1, len(result['Q_values']) + 1)
            ax.plot(K_range, result['Q_values'], 'bo-', linewidth=2)
            ax.axhline(y=result['threshold'], color='r', linestyle='--', linewidth=2,
                      label=f"Threshold: {result['threshold']:.2f}")
            ax.axvline(x=result['K'], color='g', linestyle=':', linewidth=2,
                      label=f"Optimal K={result['K']}")
            ax.set_xlabel('K')
            ax.set_ylabel('Avg Variance')
            ax.set_title('Simple Variance Threshold', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Visualize clusters
        st.subheader("Step 4: Job Cluster Visualization")
        
        jobs = result['jobs']
        labels = result['labels']
        K = result['K']
        mean_ranks = result['mean_ranks'].flatten()
        
        # Sort clusters by center
        cluster_centers = []
        for k in range(K):
            cluster_jobs_idx = [i for i in range(len(jobs)) if labels[i] == k]
            cluster_center = np.mean([mean_ranks[i] for i in cluster_jobs_idx])
            cluster_centers.append((k, cluster_center))
        cluster_centers.sort(key=lambda x: x[1])
        cluster_order = [k for k, _ in cluster_centers]
        layer_mapping = {old_k: new_k for new_k, old_k in enumerate(cluster_order)}
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, K))
        
        for layer_id in range(K):
            jobs_in_layer = [jobs[i] for i in range(len(jobs)) if layer_mapping[labels[i]] == layer_id]
            x_vals = [mean_ranks[jobs.index(j)] for j in jobs_in_layer]
            y_vals = [layer_id] * len(jobs_in_layer)
            
            ax.scatter(x_vals, y_vals, s=200, alpha=0.7, color=colors[layer_id],
                      edgecolors='black', linewidths=1.5, label=f'Layer {layer_id}')
            
            for j, x in zip(jobs_in_layer, x_vals):
                ax.text(x, layer_id, j, ha='center', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Average Position in Ranking', fontsize=12)
        ax.set_ylabel('K-means Layer', fontsize=12)
        ax.set_title(f'Job Clusters (K={K})', fontsize=14, fontweight='bold')
        ax.set_yticks(range(K))
        ax.set_yticklabels([f'Layer {i}' for i in range(K)])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, K - 0.5)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Layer details
        with st.expander("View Detailed Layer Assignments"):
            for layer_id in range(K):
                jobs_in_layer = [jobs[i] for i in range(len(jobs)) if layer_mapping[labels[i]] == layer_id]
                jobs_with_ranks = [(j, mean_ranks[jobs.index(j)]) for j in jobs_in_layer]
                jobs_with_ranks.sort(key=lambda x: x[1])
                
                center = cluster_centers[layer_id][1]
                st.write(f"**Layer {layer_id}** (center: {center:.2f})")
                for job, rank in jobs_with_ranks:
                    dept = 'Sales' if 'Sales' in job else ('Eng' if 'Eng' in job else 'CEO')
                    st.write(f"  - {job} (rank: {rank:.2f}, dept: {dept})")
    
    else:
        st.info("Configure parameters in the sidebar and click **Run Analysis** to start!")
        
        # Show example
        st.subheader("About This Tool")
        st.markdown("""
        ### Minimum Violation Ranking (MVR)
        
        The MVR algorithm infers organizational hierarchies by:
        1. **Building a directed graph** from career transitions (ALL PAIRS method)
        2. **Finding optimal rankings** that minimize violations (edges going against the ranking)
        3. **Clustering jobs** into hierarchical levels using K-means
        
        ### Four K-means Methods
        
        1. **Bonhomme et al. (2019)**: Threshold based on sum of job position variances
        2. **Overall Std**: Threshold based on variance of average ranks
        3. **Elbow Method**: Visual inspection of inertia curve (manual K selection)
        4. **Simple Variance**: Uses median job variance as threshold
        
        ### Test Case
        
        - 10 workers: Sales1 -> Sales2 -> ... -> Sales6 -> CEO
        - 10 workers: Eng1 -> Eng2 -> Eng3 -> Eng4 -> CEO
        - Theoretical max optimal rankings: C(10,6) = 210
        """)

if __name__ == "__main__":
    main()
