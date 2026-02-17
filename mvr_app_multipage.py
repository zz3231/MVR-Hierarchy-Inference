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
import mvr_algorithms as mvr

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
page = st.sidebar.radio("Navigate", ["Company Builder", "MVR Analysis", "Sensitivity Analysis"])

# ==================== PAGE 1: COMPANY BUILDER ====================

if page == "Company Builder":
    st.title("Synthetic Company Builder")
    st.markdown("""
    Build synthetic companies with configurable selection bias to test MVR algorithm robustness.
    """)
    
    st.subheader("1. Configure Firm Structure")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_departments = st.number_input("Number of Departments", 1, 6, 2)
        
        departments = []
        st.markdown("**Department Configuration:**")
        
        # Define default configurations
        default_names = ["Sales", "Engineering", "Finance", "Law", "Marketing", "HR"]
        default_ranks = [6, 4, 5, 5, 5, 5]  # Sales: 6, Engineering: 4, others: 5
        
        for i in range(n_departments):
            dept_col1, dept_col2 = st.columns([2, 1])
            with dept_col1:
                dept_name = st.text_input(f"Dept {i+1} Name", 
                                         value=default_names[i],
                                         key=f"dept_name_{i}")
            with dept_col2:
                n_ranks = st.number_input(f"Ranks", 1, 10, default_ranks[i], key=f"dept_ranks_{i}")
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
    
    # Algorithm Configuration
    st.subheader("1. Algorithm Configuration")
    
    with st.expander("Configure Algorithm Variants (Paper-Exact vs Alternative)", expanded=True):
        st.markdown("""
        Choose between paper-exact implementations (Huitfeldt et al., 2023) and alternative variants.
        
        **Paper-Exact**: Strictly follows the published methodology.  
        **Alternative**: Variants for robustness testing and comparison.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Directed Graph Construction**")
            graph_method = st.radio(
                "Method",
                ["Consecutive Pairs (Paper)", "All Pairs (Alternative)"],
                index=0,
                help="""
                Consecutive: A->B->C creates edges A->B, B->C
                All Pairs: A->B->C creates A->B, A->C, B->C
                
                Paper uses consecutive transitions only.
                """,
                key="graph_method"
            )
        
        with col2:
            st.markdown("**Initial Ranking**")
            ranking_method = st.radio(
                "Method",
                ["Unweighted Out-Degree (Paper)", "Weighted Out-Degree (Alternative)"],
                index=0,
                help="""
                Unweighted: Count number of outgoing edges
                Weighted: Sum of edge weights (transition frequencies)
                
                Paper Algorithm 2 uses unweighted out-degree.
                """,
                key="ranking_method"
            )
        
        with col3:
            st.markdown("**K-means Threshold**")
            threshold_method = st.radio(
                "Method",
                [
                    "Bonhomme Exact (Paper)",
                    "Bonhomme Scaled (Alternative)",
                    "Overall Variance Exact",
                    "Overall Variance Scaled"
                ],
                index=0,
                help="""
                Bonhomme Exact: threshold = sum(Var(r_v)) / (N-1) [PAPER]
                Bonhomme Scaled: threshold = sum(Var(r_v)) * N / (N-1)
                Overall Exact: threshold = Var(avg_ranks) / (N-1)
                Overall Scaled: threshold = Var(avg_ranks) * N / (N-1)
                
                Paper Equation B2 uses Bonhomme Exact.
                """,
                key="threshold_method"
            )
        
        # Convert display names to internal keys
        graph_key = "consecutive" if "Consecutive" in graph_method else "all_pairs"
        ranking_key = "unweighted" if "Unweighted" in ranking_method else "weighted"
        
        if "Bonhomme Exact" in threshold_method:
            threshold_key = "bonhomme_exact"
        elif "Bonhomme Scaled" in threshold_method:
            threshold_key = "bonhomme_scaled"
        elif "Overall Variance Exact" in threshold_method:
            threshold_key = "overall_exact"
        else:
            threshold_key = "overall_scaled"
        
        st.info(f"""
        **Current Configuration:**
        - Graph: {graph_key}
        - Initial Ranking: {ranking_key}
        - K-means: {threshold_key}
        """)
    
    # MVR Parameters
    st.subheader("2. MVR Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        R = st.slider("R (Repetitions)", 100, 5000, 1000, 100)
    with col2:
        T = st.slider("T (Iterations)", 100, 3000, 1000, 100)
    with col3:
        seed = st.number_input("Random Seed", 0, 10000, 42)
    with col4:
        reset_each_rep = st.checkbox(
            "Reset Each R", 
            value=True,
            help="Paper: Reset to initial ranking at each R repetition (independent searches). Alternative: Continue from previous R (cumulative search)."
        )
    
    with st.expander("Understanding R Repetition Strategy"):
        st.markdown("""
        **Reset Each R (Paper Method, Default)**: ✅
        - Each R repetition starts fresh from initial ranking
        - R independent searches of solution space
        - Better at finding multiple global optima
        - Paper: "each time starting with the same initial ranking"
        
        **Continue from Previous R (Alternative)**:
        - Each R repetition continues from where previous R ended
        - One long cumulative search
        - May converge faster to single optimum
        - Explores fewer diverse solutions
        
        **Recommendation**: Use paper method (Reset=True) for standard analysis.
        """)
    
    # ILM Pruning Settings
    st.subheader("3. ILM Network Pruning")
    
    col1, col2 = st.columns(2)
    with col1:
        enable_pruning = st.checkbox("Enable ILM Pruning (Paper's Algorithm 1)", value=True,
                                     help="Remove articulation points to address measurement error")
    with col2:
        X_threshold = st.slider("X% Threshold", min_value=5, max_value=20, value=10, step=5,
                               help="Remove workers affecting less than X% of jobs")
    
    if st.button("Run MVR Analysis on Both Companies", type="primary"):
        
        # Create progress callbacks
        def progress_callback_gt(current, total):
            pass  # Progress handled by streamlit progress bar
        
        def progress_callback_bias(current, total):
            pass
        
        # Run Ground Truth Company
        with st.spinner("Running MVR on Ground Truth Company..."):
            try:
                results_gt = mvr.run_complete_mvr_pipeline(
                    panel_df=st.session_state.ground_truth_df,
                    enable_ilm_pruning=enable_pruning,
                    X_threshold=X_threshold,
                    graph_method=graph_key,
                    ranking_method=ranking_key,
                    threshold_method=threshold_key,
                    reset_each_rep=reset_each_rep,
                    R=R,
                    T=T,
                    seed=seed,
                    progress_callback=progress_callback_gt
                )
                
                st.success(f"Ground Truth: {results_gt['directed_graph_nodes']} jobs, "
                          f"{results_gt['directed_graph_edges']} edges, "
                          f"{len(results_gt['optimal_rankings'])} optimal rankings, "
                          f"{results_gt['min_violations']} violations")
            except Exception as e:
                st.error(f"Error in Ground Truth analysis: {str(e)}")
                st.stop()
        
        # Run Biased Company
        with st.spinner("Running MVR on Biased Company..."):
            try:
                results_bias = mvr.run_complete_mvr_pipeline(
                    panel_df=st.session_state.biased_df,
                    enable_ilm_pruning=enable_pruning,
                    X_threshold=X_threshold,
                    graph_method=graph_key,
                    ranking_method=ranking_key,
                    threshold_method=threshold_key,
                    reset_each_rep=reset_each_rep,
                    R=R,
                    T=T,
                    seed=seed + 1,  # Different seed for biased
                    progress_callback=progress_callback_bias
                )
                
                st.success(f"Biased: {results_bias['directed_graph_nodes']} jobs, "
                          f"{results_bias['directed_graph_edges']} edges, "
                          f"{len(results_bias['optimal_rankings'])} optimal rankings, "
                          f"{results_bias['min_violations']} violations")
            except Exception as e:
                st.error(f"Error in Biased analysis: {str(e)}")
                st.stop()
        
        K_gt = results_gt['K']
        K_bias = results_bias['K']
        
        # Store results in session state
        st.session_state.mvr_results = {
            'results_gt': results_gt,
            'results_bias': results_bias,
            'K_gt': K_gt,
            'K_bias': K_bias,
            'config': {
                'graph_method': graph_key,
                'ranking_method': ranking_key,
                'threshold_method': threshold_key,
                'enable_pruning': enable_pruning,
                'X_threshold': X_threshold,
                'R': R,
                'T': T
            }
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
        
        # Extract key objects for visualization
        H_gt = results_gt['directed_graph']
        H_bias = results_bias['directed_graph']
        opt_gt = results_gt['optimal_rankings']
        opt_bias = results_bias['optimal_rankings']
        viol_gt = results_gt['min_violations']
        viol_bias = results_bias['min_violations']
        prog_gt = results_gt['progress']
        prog_bias = results_bias['progress']
        result_gt = {k: results_gt[k] for k in ['K', 'threshold', 'Q_values', 'positions', 'labels']}
        result_bias = {k: results_bias[k] for k in ['K', 'threshold', 'Q_values', 'positions', 'labels']}
        
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
        ax1.set_xlabel('Average Rank (+/- Std Dev)', fontsize=11)
        ax1.set_title('Ground Truth Company Rankings', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Biased company ranking
        means_bias = [np.mean(positions_bias[j]) for j in jobs_bias]
        stds_bias = [np.std(positions_bias[j], ddof=1) if len(positions_bias[j]) > 1 else 0 for j in jobs_bias]
        
        ax2.barh(range(len(jobs_bias)), means_bias, xerr=stds_bias, capsize=5, alpha=0.7, color='coral')
        ax2.set_yticks(range(len(jobs_bias)))
        ax2.set_yticklabels(jobs_bias)
        ax2.set_xlabel('Average Rank (+/- Std Dev)', fontsize=11)
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
            ax.set_xlabel('Average Rank (+/- Std Dev)')
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
        
        # Create method description based on threshold_key
        method_descriptions = {
            "bonhomme_exact": "Bonhomme Exact (Paper)",
            "bonhomme_scaled": "Bonhomme Scaled (Alternative)",
            "overall_exact": "Overall Variance Exact",
            "overall_scaled": "Overall Variance Scaled"
        }
        
        method_name = method_descriptions.get(threshold_key, threshold_key)
        st.info(f"Method: {method_name} | Optimal K: {K_gt} | Threshold: {result_gt['threshold']:.4f}")
        
        # Add explanation of the method
        with st.expander("Understanding K-means Methods"):
            st.markdown("""
            **Bonhomme Exact (Paper Method)**:
            - Formula: `threshold = sum(Var(r_v)) / (N-1)`
            - Paper Reference: Appendix B.3, Equation B2
            - Uses sum of individual job variances
            - Tends to produce MORE layers (larger K)
            
            **Bonhomme Scaled (Alternative)**:
            - Formula: `threshold = sum(Var(r_v)) * N / (N-1)`
            - Multiplies by N (number of jobs)
            - Tends to produce FEWER layers (smaller K)
            
            **Overall Variance Exact**:
            - Uses variance of consensus ranks: `threshold = Var(avg_ranks) / (N-1)`
            - Alternative method for comparison
            
            **Overall Variance Scaled**:
            - Scaled version: `threshold = Var(avg_ranks) * N / (N-1)`
            - Alternative method with N multiplication
            
            **Key Difference**: 
            - Paper-exact (Bonhomme Exact) has NO N multiplication
            - Scaled versions multiply by N, increasing threshold
            - Higher threshold → easier to satisfy Q(K) ≤ threshold → fewer clusters (smaller K)
            """)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        K_range = range(1, len(result_gt['Q_values']) + 1)
        ax.plot(K_range, result_gt['Q_values'], 'bo-', linewidth=2)
        ax.axhline(y=result_gt['threshold'], color='r', linestyle='--', linewidth=2,
                  label=f"Threshold: {result_gt['threshold']:.2f}")
        ax.axvline(x=K_gt, color='g', linestyle=':', linewidth=2, label=f"Optimal K={K_gt}")
        ax.set_xlabel('K')
        ax.set_ylabel('Q(K)')
        ax.set_title(f'{method_name}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
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
            ax.set_xlabel('Average Rank (+/- Std Dev)')
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
        method_name = method_descriptions.get(threshold_key, threshold_key)
        st.info(f"Method: {method_name} | Optimal K: {K_bias} | Threshold: {result_bias['threshold']:.4f}")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        K_range = range(1, len(result_bias['Q_values']) + 1)
        ax.plot(K_range, result_bias['Q_values'], 'bo-', linewidth=2)
        ax.axhline(y=result_bias['threshold'], color='r', linestyle='--', linewidth=2,
                  label=f"Threshold: {result_bias['threshold']:.2f}")
        ax.axvline(x=K_bias, color='g', linestyle=':', linewidth=2, label=f"Optimal K={K_bias}")
        ax.set_xlabel('K')
        ax.set_ylabel('Q(K)')
        ax.set_title(f'{method_name}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Algorithm Configuration Summary
        st.markdown("---")
        st.subheader("Algorithm Configuration Summary")
        
        config = st.session_state.mvr_results['config']
        
        config_df = pd.DataFrame([
            {'Component': 'Directed Graph', 'Method': config['graph_method']},
            {'Component': 'Initial Ranking', 'Method': config['ranking_method']},
            {'Component': 'K-means Threshold', 'Method': config['threshold_method']},
            {'Component': 'ILM Pruning', 'Method': f"{'Enabled' if config['enable_pruning'] else 'Disabled'} (X={config['X_threshold']}%)"},
            {'Component': 'MVR Parameters', 'Method': f"R={config['R']}, T={config['T']}"}
        ])
        
        st.dataframe(config_df, use_container_width=True, hide_index=True)
        
        # Pipeline Statistics
        st.markdown("**Pipeline Statistics**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ground Truth**")
            stats_gt = pd.DataFrame([
                {'Stage': 'Initial Graph', 'Value': f"{results_gt['initial_jobs']} jobs, {results_gt['initial_workers']} workers"},
                {'Stage': 'Workers Removed', 'Value': f"{results_gt['workers_removed']}"},
                {'Stage': 'Largest ILM', 'Value': f"{results_gt['largest_ilm_jobs']} jobs, {results_gt['largest_ilm_workers']} workers"},
                {'Stage': 'Directed Graph', 'Value': f"{results_gt['directed_graph_nodes']} nodes, {results_gt['directed_graph_edges']} edges"},
                {'Stage': 'Optimal Rankings', 'Value': f"{len(results_gt['optimal_rankings'])} rankings"},
                {'Stage': 'Min Violations', 'Value': f"{results_gt['min_violations']}"},
                {'Stage': 'Final K', 'Value': f"{K_gt}"}
            ])
            st.dataframe(stats_gt, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Biased Company**")
            stats_bias = pd.DataFrame([
                {'Stage': 'Initial Graph', 'Value': f"{results_bias['initial_jobs']} jobs, {results_bias['initial_workers']} workers"},
                {'Stage': 'Workers Removed', 'Value': f"{results_bias['workers_removed']}"},
                {'Stage': 'Largest ILM', 'Value': f"{results_bias['largest_ilm_jobs']} jobs, {results_bias['largest_ilm_workers']} workers"},
                {'Stage': 'Directed Graph', 'Value': f"{results_bias['directed_graph_nodes']} nodes, {results_bias['directed_graph_edges']} edges"},
                {'Stage': 'Optimal Rankings', 'Value': f"{len(results_bias['optimal_rankings'])} rankings"},
                {'Stage': 'Min Violations', 'Value': f"{results_bias['min_violations']}"},
                {'Stage': 'Final K', 'Value': f"{K_bias}"}
            ])
            st.dataframe(stats_bias, use_container_width=True, hide_index=True)

# ==================== PAGE 3: SENSITIVITY ANALYSIS ====================

elif page == "Sensitivity Analysis":
    st.title("Sensitivity Analysis: Leave-One-Job-Out (LOJO)")
    st.markdown("""
    Analyze algorithm sensitivity by systematically removing each job and comparing results.
    
    This helps identify:
    - Which jobs are critical for hierarchy identification
    - How rank stability varies across jobs
    - Robustness of different algorithm configurations
    """)
    
    # Check prerequisites
    if st.session_state.ground_truth_df is None:
        st.warning("Please generate a Ground Truth company in Page 1 first.")
        st.stop()
    
    if st.session_state.firm_structure is None:
        st.warning("Please configure firm structure in Page 1 first.")
        st.stop()
    
    # Display current company structure
    st.info(f"Current Company: {len(st.session_state.ground_truth_df)} records, "
            f"{st.session_state.ground_truth_df['worker_id'].nunique()} workers, "
            f"{st.session_state.ground_truth_df['role'].nunique()} jobs")
    
    # Get all jobs
    all_jobs = sorted(st.session_state.ground_truth_df['role'].unique())
    n_jobs = len(all_jobs)
    
    st.markdown(f"**Total scenarios to test**: {n_jobs + 1} (Full data + {n_jobs} single-job-missing scenarios)")
    
    # Algorithm Configuration
    st.subheader("1. Algorithm Configuration")
    
    with st.expander("Select Algorithm Variants to Compare", expanded=True):
        st.markdown("""
        Choose which algorithm configurations to test.
        Each configuration will be tested on all scenarios.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Directed Graph**")
            test_consecutive = st.checkbox("Consecutive Pairs (Paper)", value=True, key="sens_consecutive")
            test_all_pairs = st.checkbox("All Pairs (Alternative)", value=True, key="sens_all_pairs")
        
        with col2:
            st.markdown("**Initial Ranking**")
            test_unweighted = st.checkbox("Unweighted Out-Degree (Paper)", value=True, key="sens_unweighted")
            test_weighted = st.checkbox("Weighted Out-Degree (Alternative)", value=False, key="sens_weighted")
        
        with col3:
            st.markdown("**K-means Threshold**")
            test_bonhomme_exact = st.checkbox("Bonhomme Exact (Paper)", value=True, key="sens_bonhomme_exact")
            test_bonhomme_scaled = st.checkbox("Bonhomme Scaled", value=False, key="sens_bonhomme_scaled")
        
        # Calculate total combinations
        graph_methods = []
        if test_consecutive:
            graph_methods.append("consecutive")
        if test_all_pairs:
            graph_methods.append("all_pairs")
        
        ranking_methods = []
        if test_unweighted:
            ranking_methods.append("unweighted")
        if test_weighted:
            ranking_methods.append("weighted")
        
        threshold_methods = []
        if test_bonhomme_exact:
            threshold_methods.append("bonhomme_exact")
        if test_bonhomme_scaled:
            threshold_methods.append("bonhomme_scaled")
        
        n_configs = len(graph_methods) * len(ranking_methods) * len(threshold_methods)
        
        if n_configs == 0:
            st.error("Please select at least one option from each category.")
            st.stop()
        
        st.info(f"**Selected {n_configs} configuration(s)** to test on {n_jobs + 1} scenarios = **{n_configs * (n_jobs + 1)} total runs**")
    
    # MVR Parameters
    st.subheader("2. MVR Parameters")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        R = st.slider("R (Repetitions)", 100, 2000, 500, 100, key="sens_R")
    with col2:
        T = st.slider("T (Iterations)", 100, 2000, 500, 100, key="sens_T")
    with col3:
        seed = st.number_input("Random Seed", 0, 10000, 42, key="sens_seed")
    with col4:
        reset_each_rep = st.checkbox("Reset Each R", value=True, key="sens_reset_each_rep",
                                     help="Reset to initial ranking at each R (paper method)")
    with col5:
        enable_cache = st.checkbox("Cache Results", value=True, help="Cache results to avoid re-computation")
    
    # ILM Pruning
    st.subheader("3. ILM Network Pruning")
    
    col1, col2 = st.columns(2)
    with col1:
        enable_pruning = st.checkbox("Enable ILM Pruning", value=False, key="sens_enable_pruning")
    with col2:
        if enable_pruning:
            X_threshold = st.slider("X Threshold (%)", 5, 20, 10, 1, key="sens_X_threshold")
        else:
            X_threshold = 10
    
    # Initialize session state for caching
    if 'sensitivity_cache' not in st.session_state:
        st.session_state.sensitivity_cache = {}
    
    # Run Analysis Button
    if st.button("Run Sensitivity Analysis", type="primary"):
        
        # Create all configurations
        configs = []
        for graph_method in graph_methods:
            for ranking_method in ranking_methods:
                for threshold_method in threshold_methods:
                    configs.append({
                        'graph_method': graph_method,
                        'ranking_method': ranking_method,
                        'threshold_method': threshold_method,
                        'enable_pruning': enable_pruning,
                        'X_threshold': X_threshold,
                        'reset_each_rep': reset_each_rep,
                        'R': R,
                        'T': T,
                        'seed': seed
                    })
        
        # Create cache key
        cache_key = str(configs) + str(all_jobs)
        
        # Check cache
        if enable_cache and cache_key in st.session_state.sensitivity_cache:
            st.info("Loading cached results...")
            all_results = st.session_state.sensitivity_cache[cache_key]
        else:
            # Run analysis
            all_results = {}
            
            total_runs = len(configs) * (n_jobs + 1)
            progress_bar = st.progress(0)
            status_text = st.empty()
            current_run = 0
            
            for config_idx, config in enumerate(configs):
                config_name = f"{config['graph_method']}_{config['ranking_method']}_{config['threshold_method']}"
                
                status_text.text(f"Testing configuration {config_idx + 1}/{len(configs)}: {config_name}")
                
                config_results = []
                
                # Scenario 0: Full data (no missing job)
                status_text.text(f"Config {config_idx + 1}/{len(configs)}: Running full data scenario...")
                
                try:
                    result_full = mvr.run_complete_mvr_pipeline(
                        panel_df=st.session_state.ground_truth_df,
                        enable_ilm_pruning=config['enable_pruning'],
                        X_threshold=config['X_threshold'],
                        graph_method=config['graph_method'],
                        ranking_method=config['ranking_method'],
                        threshold_method=config['threshold_method'],
                        reset_each_rep=config['reset_each_rep'],
                        R=config['R'],
                        T=config['T'],
                        seed=config['seed']
                    )
                    
                    config_results.append({
                        'missing_job': 'None (Full Data)',
                        'K': result_full['K'],
                        'violations': result_full['min_violations'],
                        'n_optimal_rankings': len(result_full['optimal_rankings']),
                        'positions': result_full['positions'],
                        'labels': result_full['labels'],
                        'threshold': result_full['threshold']
                    })
                except Exception as e:
                    st.error(f"Error in full data scenario: {str(e)}")
                    config_results.append({
                        'missing_job': 'None (Full Data)',
                        'K': None,
                        'violations': None,
                        'n_optimal_rankings': None,
                        'positions': {},
                        'labels': {},
                        'threshold': None,
                        'error': str(e)
                    })
                
                current_run += 1
                progress_bar.progress(current_run / total_runs)
                
                # Scenarios 1-N: Each job missing
                for job_idx, missing_job in enumerate(all_jobs):
                    status_text.text(f"Config {config_idx + 1}/{len(configs)}: Testing without {missing_job} ({job_idx + 1}/{n_jobs})...")
                    
                    # Create scenario data (remove all records with this job)
                    scenario_df = st.session_state.ground_truth_df[
                        st.session_state.ground_truth_df['role'] != missing_job
                    ].copy()
                    
                    try:
                        result = mvr.run_complete_mvr_pipeline(
                            panel_df=scenario_df,
                            enable_ilm_pruning=config['enable_pruning'],
                            X_threshold=config['X_threshold'],
                            graph_method=config['graph_method'],
                            ranking_method=config['ranking_method'],
                            threshold_method=config['threshold_method'],
                            reset_each_rep=config['reset_each_rep'],
                            R=config['R'],
                            T=config['T'],
                            seed=config['seed'] + job_idx + 1
                        )
                        
                        config_results.append({
                            'missing_job': missing_job,
                            'K': result['K'],
                            'violations': result['min_violations'],
                            'n_optimal_rankings': len(result['optimal_rankings']),
                            'positions': result['positions'],
                            'labels': result['labels'],
                            'threshold': result['threshold']
                        })
                    except Exception as e:
                        st.error(f"Error with {missing_job} missing: {str(e)}")
                        config_results.append({
                            'missing_job': missing_job,
                            'K': None,
                            'violations': None,
                            'n_optimal_rankings': None,
                            'positions': {},
                            'labels': {},
                            'threshold': None,
                            'error': str(e)
                        })
                    
                    current_run += 1
                    progress_bar.progress(current_run / total_runs)
                
                all_results[config_name] = config_results
            
            progress_bar.empty()
            status_text.empty()
            
            # Cache results
            if enable_cache:
                st.session_state.sensitivity_cache[cache_key] = all_results
            
            st.success(f"Completed {total_runs} runs!")
        
        # Store results in session state
        st.session_state.sensitivity_results = {
            'all_results': all_results,
            'all_jobs': all_jobs,
            'configs': configs,
            'config_names': list(all_results.keys())
        }
    
    # Display Results
    if 'sensitivity_results' in st.session_state:
        st.markdown("---")
        st.header("Results")
        
        results_data = st.session_state.sensitivity_results
        all_results = results_data['all_results']
        all_jobs = results_data['all_jobs']
        config_names = results_data['config_names']
        
        # Configuration selector for detailed view
        if len(config_names) > 1:
            selected_config = st.selectbox("Select Configuration to View", config_names)
        else:
            selected_config = config_names[0]
        
        config_results = all_results[selected_config]
        
        # Extract data for this configuration
        full_result = config_results[0]
        job_results = config_results[1:]
        
        # TABLE 1: Optimal Average Ranks
        st.subheader("Table 1: Optimal Average Ranks")
        st.markdown("Average rank position of each job across all optimal rankings in each scenario.")
        
        rank_data = []
        for result in config_results:
            row = {'Scenario': result['missing_job']}
            positions = result.get('positions', {})
            for job in all_jobs:
                if job in positions and len(positions[job]) > 0:
                    row[job] = f"{np.mean(positions[job]):.2f}"
                else:
                    row[job] = "-"
            rank_data.append(row)
        
        rank_df = pd.DataFrame(rank_data)
        st.dataframe(rank_df, use_container_width=True, height=400)
        
        # TABLE 2: Rank Standard Deviation
        st.subheader("Table 2: Rank Standard Deviation (Within Optimal Rankings)")
        st.markdown("Standard deviation of rank position across optimal rankings for each job.")
        
        std_data = []
        for result in config_results:
            row = {'Scenario': result['missing_job']}
            positions = result.get('positions', {})
            for job in all_jobs:
                if job in positions and len(positions[job]) > 1:
                    row[job] = f"{np.std(positions[job], ddof=1):.3f}"
                elif job in positions and len(positions[job]) == 1:
                    row[job] = "0.000"
                else:
                    row[job] = "-"
            std_data.append(row)
        
        std_df = pd.DataFrame(std_data)
        st.dataframe(std_df, use_container_width=True, height=400)
        
        # TABLE 3: K Values and Violations
        st.subheader("Table 3: Identified K Values")
        st.markdown("Number of hierarchy levels identified by K-means clustering.")
        
        K_full = full_result['K']
        
        k_data = []
        for result in config_results:
            K_identified = result.get('K', None)
            violations = result.get('violations', None)
            n_rankings = result.get('n_optimal_rankings', None)
            
            k_data.append({
                'Scenario': result['missing_job'],
                'K_identified': K_identified if K_identified is not None else "Error",
                'K_full': K_full,
                'Correct': "✓" if K_identified == K_full else "✗",
                'Delta': K_identified - K_full if K_identified is not None else "-",
                'Violations': violations if violations is not None else "-",
                'N_Optimal_Rankings': n_rankings if n_rankings is not None else "-"
            })
        
        k_df = pd.DataFrame(k_data)
        
        # Highlight critical jobs (where K changes)
        def highlight_critical(row):
            if row['Correct'] == "✗":
                return ['background-color: #ffcccc'] * len(row)
            elif row['Scenario'] == 'None (Full Data)':
                return ['background-color: #ccffcc'] * len(row)
            return [''] * len(row)
        
        styled_k_df = k_df.style.apply(highlight_critical, axis=1)
        st.dataframe(styled_k_df, use_container_width=True, height=400)
        
        # Summary metrics
        n_correct = sum(1 for item in k_data[1:] if item['Correct'] == "✓")
        accuracy = n_correct / len(job_results) * 100 if job_results else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("K Recovery Rate", f"{accuracy:.1f}%", 
                     help="Percentage of scenarios where K was correctly identified")
        with col2:
            critical_jobs = [item['Scenario'] for item in k_data[1:] if item['Correct'] == "✗"]
            st.metric("Critical Jobs", len(critical_jobs),
                     help="Jobs whose absence causes K to change")
        with col3:
            st.metric("Full Data K", K_full)
        
        if critical_jobs:
            st.warning(f"**Critical jobs identified**: {', '.join(critical_jobs)}")
        
        # Visualizations
        st.markdown("---")
        st.subheader("Visualizations")
        
        # Visualization 1: K Stability Plot
        st.markdown("**K Value by Missing Job**")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        scenarios = [item['Scenario'] for item in k_data]
        K_values = [item['K_identified'] if isinstance(item['K_identified'], (int, float)) else None for item in k_data]
        colors = ['green' if item['Correct'] == "✓" else 'red' if item['Correct'] == "✗" else 'gray' for item in k_data]
        
        x_pos = range(len(scenarios))
        ax.scatter(x_pos, K_values, c=colors, s=100, alpha=0.6, edgecolors='black')
        ax.axhline(y=K_full, color='blue', linestyle='--', linewidth=2, label=f'Full Data K={K_full}')
        
        ax.set_xlabel('Scenario', fontweight='bold')
        ax.set_ylabel('Identified K', fontweight='bold')
        ax.set_title(f'K Stability Across Scenarios ({selected_config})', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Visualization 2: Rank Heatmap
        st.markdown("**Rank Position Heatmap**")
        st.markdown("Shows average rank of each job (columns) in each scenario (rows). Missing jobs shown as white.")
        
        # Create heatmap data
        heatmap_data = np.zeros((len(config_results), len(all_jobs)))
        heatmap_data[:] = np.nan
        
        for i, result in enumerate(config_results):
            positions = result.get('positions', {})
            for j, job in enumerate(all_jobs):
                if job in positions and len(positions[job]) > 0:
                    heatmap_data[i, j] = np.mean(positions[job])
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(config_results) * 0.4)))
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
        
        ax.set_xticks(range(len(all_jobs)))
        ax.set_xticklabels(all_jobs, rotation=45, ha='right')
        ax.set_yticks(range(len(config_results)))
        ax.set_yticklabels([r['missing_job'] for r in config_results])
        
        ax.set_xlabel('Job', fontweight='bold')
        ax.set_ylabel('Scenario (Missing Job)', fontweight='bold')
        ax.set_title(f'Average Rank Heatmap ({selected_config})', fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Rank', rotation=270, labelpad=20)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Visualization 3: Rank Stability (Std Dev) Bar Chart
        st.markdown("**Rank Stability Analysis**")
        st.markdown("Jobs with higher standard deviation have less stable rankings.")
        
        # Calculate average std across all scenarios for each job
        job_std_avg = {}
        for job in all_jobs:
            stds = []
            for result in config_results:
                positions = result.get('positions', {})
                if job in positions and len(positions[job]) > 1:
                    stds.append(np.std(positions[job], ddof=1))
            job_std_avg[job] = np.mean(stds) if stds else 0
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        jobs_sorted = sorted(job_std_avg.keys(), key=lambda x: job_std_avg[x], reverse=True)
        stds_sorted = [job_std_avg[j] for j in jobs_sorted]
        
        bars = ax.bar(range(len(jobs_sorted)), stds_sorted, color='steelblue', alpha=0.7)
        
        # Highlight high-variance jobs
        threshold_high = np.mean(stds_sorted) + np.std(stds_sorted)
        for i, (job, std) in enumerate(zip(jobs_sorted, stds_sorted)):
            if std > threshold_high:
                bars[i].set_color('red')
        
        ax.set_xlabel('Job', fontweight='bold')
        ax.set_ylabel('Average Rank Std Dev', fontweight='bold')
        ax.set_title(f'Rank Stability by Job ({selected_config})', fontweight='bold')
        ax.set_xticks(range(len(jobs_sorted)))
        ax.set_xticklabels(jobs_sorted, rotation=45, ha='right')
        ax.axhline(y=np.mean(stds_sorted), color='gray', linestyle='--', alpha=0.5, label='Mean')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Configuration Comparison (if multiple configs tested)
        if len(config_names) > 1:
            st.markdown("---")
            st.subheader("Configuration Comparison")
            st.markdown("Compare K recovery rates and critical jobs across different algorithm configurations.")
            
            comparison_data = []
            for config_name in config_names:
                config_results = all_results[config_name]
                full_K = config_results[0]['K']
                job_results = config_results[1:]
                
                n_correct = sum(1 for r in job_results if r.get('K') == full_K)
                accuracy = n_correct / len(job_results) * 100 if job_results else 0
                critical_jobs = [r['missing_job'] for r in job_results if r.get('K') != full_K]
                
                comparison_data.append({
                    'Configuration': config_name,
                    'Full_Data_K': full_K,
                    'K_Recovery_Rate': f"{accuracy:.1f}%",
                    'Critical_Jobs_Count': len(critical_jobs),
                    'Critical_Jobs': ', '.join(critical_jobs) if critical_jobs else 'None'
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Bar chart comparing recovery rates
            fig, ax = plt.subplots(figsize=(10, 5))
            
            configs = [item['Configuration'] for item in comparison_data]
            rates = [float(item['K_Recovery_Rate'].rstrip('%')) for item in comparison_data]
            
            bars = ax.bar(range(len(configs)), rates, color='steelblue', alpha=0.7)
            
            # Color code by performance
            for i, rate in enumerate(rates):
                if rate >= 90:
                    bars[i].set_color('green')
                elif rate >= 70:
                    bars[i].set_color('orange')
                else:
                    bars[i].set_color('red')
            
            ax.set_xlabel('Configuration', fontweight='bold')
            ax.set_ylabel('K Recovery Rate (%)', fontweight='bold')
            ax.set_title('Algorithm Robustness Comparison', fontweight='bold')
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, rotation=45, ha='right')
            ax.set_ylim(0, 100)
            ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Export Results
        st.markdown("---")
        st.subheader("Export Results")
        
        if st.button("Export All Tables to CSV"):
            import io
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                rank_df.to_excel(writer, sheet_name='Average_Ranks', index=False)
                std_df.to_excel(writer, sheet_name='Rank_StdDev', index=False)
                k_df.to_excel(writer, sheet_name='K_Values', index=False)
                if len(config_names) > 1:
                    comparison_df.to_excel(writer, sheet_name='Config_Comparison', index=False)
            
            buffer.seek(0)
            
            st.download_button(
                label="Download Excel File",
                data=buffer,
                file_name=f"sensitivity_analysis_{selected_config}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    pass
