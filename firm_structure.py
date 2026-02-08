"""
Firm structure utilities for synthetic company generation.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import defaultdict


class FirmStructure:
    def __init__(self, departments_config):
        """
        Initialize firm structure.
        
        Args:
            departments_config: List of (dept_name, n_ranks) tuples
                Example: [("Law", 6), ("Engineering", 7), ("Sales", 4)]
        """
        self.departments_config = departments_config
        self.max_rank = max(n_ranks for _, n_ranks in departments_config)
        self.roles = self._create_roles()
        
    def _create_roles(self):
        """Create all roles in the firm."""
        roles = {}
        
        for dept, n_ranks in self.departments_config:
            dept_roles = []
            for rank in range(1, n_ranks + 1):
                role_name = f"{dept}_L{rank}"
                dept_roles.append((role_name, rank))
            roles[dept] = dept_roles
        
        roles["Executive"] = [("CEO", self.max_rank + 1)]
        return roles
    
    def get_all_roles_df(self):
        """Return DataFrame of all roles."""
        data = []
        for dept, role_list in self.roles.items():
            for role_name, rank in role_list:
                data.append({
                    'department': dept,
                    'role': role_name,
                    'rank': rank
                })
        return pd.DataFrame(data)
    
    def visualize_structure(self):
        """Visualize firm structure with vertical columns per department."""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        dept_colors = {
            'Law': '#1f77b4',
            'Engineering': '#ff7f0e',
            'Sales': '#2ca02c',
            'Finance': '#d62728',
            'Executive': '#9467bd'
        }
        
        non_exec_depts = [dept for dept in self.roles.keys() if dept != 'Executive']
        dept_x_positions = {dept: i * 3 for i, dept in enumerate(non_exec_depts)}
        
        max_rank = 0
        
        for dept in non_exec_depts:
            role_list = self.roles[dept]
            x_pos = dept_x_positions[dept]
            color = dept_colors.get(dept, 'gray')
            
            for role_name, rank in role_list:
                max_rank = max(max_rank, rank)
                y_pos = rank
                
                ax.scatter(x_pos, y_pos, s=800, c=color, alpha=0.7, 
                          edgecolors='black', linewidths=2, zorder=3)
                ax.text(x_pos, y_pos, role_name.replace('_', '\n'), 
                       ha='center', va='center', fontsize=8, fontweight='bold')
                
                if rank > 1:
                    ax.plot([x_pos, x_pos], [rank - 0.3, rank - 0.7], 
                           'k-', linewidth=1.5, alpha=0.5, zorder=1)
            
            top_rank = max(r for _, r in role_list)
            ceo_rank = self.roles['Executive'][0][1]
            ax.plot([x_pos, x_pos], [top_rank + 0.3, ceo_rank - 0.3], 
                   'k--', linewidth=1.5, alpha=0.5, zorder=1)
            
            ax.scatter(x_pos, ceo_rank, s=1000, c=dept_colors['Executive'], alpha=0.8, 
                      edgecolors='black', linewidths=2.5, zorder=4)
            ax.text(x_pos, ceo_rank, 'CEO', ha='center', va='center', 
                   fontsize=9, fontweight='bold')
        
        for dept, x_pos in dept_x_positions.items():
            ax.text(x_pos, 0.2, dept, ha='center', va='top', 
                   fontsize=11, fontweight='bold', 
                   color=dept_colors.get(dept, 'black'))
        
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=dept_colors.get(dept, 'gray'), 
                                      markersize=10, label=dept)
                          for dept in non_exec_depts]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=dept_colors['Executive'], 
                                          markersize=12, label='CEO'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.set_ylabel('Rank Level', fontsize=12, fontweight='bold')
        ax.set_title('Firm Hierarchy Structure (Ground Truth)', 
                     fontsize=14, fontweight='bold')
        ax.set_yticks(range(1, max_rank + 2))
        ax.set_ylim(0, max_rank + 2)
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        
        plt.tight_layout()
        return fig


class PromotionConfig:
    def __init__(self, firm_structure, default_prob=0.20):
        """Initialize promotion probabilities."""
        self.firm = firm_structure
        self.default_prob = default_prob
        self.probs = self._init_default_probs()
    
    def _init_default_probs(self):
        """Initialize all roles with default promotion probability."""
        probs = {}
        for dept, role_list in self.firm.roles.items():
            for role_name, rank in role_list:
                if role_name == "CEO":
                    probs[role_name] = 0.0
                else:
                    probs[role_name] = self.default_prob
        return probs
    
    def set_promotion_prob(self, role_name, prob):
        """Set custom promotion probability for a specific role."""
        if role_name in self.probs:
            self.probs[role_name] = prob
        else:
            raise ValueError(f"Role {role_name} not found")
    
    def set_department_prob(self, dept_name, prob):
        """Set promotion probability for all roles in a department."""
        if dept_name in self.firm.roles:
            for role_name, _ in self.firm.roles[dept_name]:
                if role_name != "CEO":
                    self.probs[role_name] = prob
        else:
            raise ValueError(f"Department {dept_name} not found")


class WorkerGenerator:
    def __init__(self, firm_structure, promotion_config, 
                 n_workers=500, start_year=2016, end_year=2023, seed=42):
        """Generate worker trajectories."""
        self.firm = firm_structure
        self.promo_config = promotion_config
        self.n_workers = n_workers
        self.start_year = start_year
        self.end_year = end_year
        random.seed(seed)
        np.random.seed(seed)
        
    def generate_trajectories(self):
        """Generate complete worker trajectories."""
        trajectories = []
        worker_id = 1
        depts = [d for d in self.firm.roles.keys() if d != 'Executive']
        
        for _ in range(self.n_workers):
            dept = random.choice(depts)
            dept_roles = self.firm.roles[dept]
            current_role, current_rank = dept_roles[0]
            
            for year in range(self.start_year, self.end_year + 1):
                trajectories.append({
                    'worker_id': worker_id,
                    'year': year,
                    'department': dept,
                    'role': current_role,
                    'rank': current_rank
                })
                
                promo_prob = self.promo_config.probs[current_role]
                if random.random() < promo_prob:
                    next_rank = current_rank + 1
                    if next_rank > len(dept_roles):
                        current_role = 'CEO'
                        current_rank = self.firm.roles['Executive'][0][1]
                        dept = 'Executive'
                    else:
                        for role_name, rank in dept_roles:
                            if rank == next_rank:
                                current_role = role_name
                                current_rank = rank
                                break
            worker_id += 1
        
        return pd.DataFrame(trajectories)
    
    def get_summary_table(self, trajectories_df):
        """Get summary: average employee count per role."""
        n_years = len(range(self.start_year, self.end_year + 1))
        avg_counts = trajectories_df.groupby(['department', 'role', 'rank']).size() / n_years
        return avg_counts.reset_index(name='avg_employees').sort_values(['department', 'rank'])


class ObservationBiasSimulator:
    def __init__(self, ground_truth_df, seed=42):
        """Simulate observation bias."""
        self.ground_truth = ground_truth_df.copy()
        self.observation_rates = {}
        random.seed(seed)
        np.random.seed(seed)
        
    def set_observation_rate(self, role, rate):
        """Set observation rate for a specific role."""
        if not 0 <= rate <= 1:
            raise ValueError("Rate must be between 0 and 1")
        self.observation_rates[role] = rate
    
    def set_layer_based_rates(self, rates_by_rank):
        """Set observation rates based on rank level."""
        for _, row in self.ground_truth[['role', 'rank']].drop_duplicates().iterrows():
            role, rank = row['role'], row['rank']
            rate = rates_by_rank.get(rank, rates_by_rank[max(rates_by_rank.keys())])
            self.observation_rates[role] = rate
    
    def apply_bias(self):
        """Apply observation bias to create biased sample."""
        biased_rows = []
        for _, row in self.ground_truth.iterrows():
            if random.random() < self.observation_rates.get(row['role'], 1.0):
                biased_rows.append(row)
        return pd.DataFrame(biased_rows)
    
    def get_comparison_table(self, biased_df):
        """Compare ground truth vs biased sample."""
        n_years = len(self.ground_truth['year'].unique())
        
        gt_counts = self.ground_truth.groupby(['department', 'role', 'rank']).size() / n_years
        gt_counts = gt_counts.reset_index(name='ground_truth_avg')
        
        if len(biased_df) > 0:
            biased_counts = biased_df.groupby(['department', 'role', 'rank']).size() / n_years
            biased_counts = biased_counts.reset_index(name='observed_avg')
        else:
            biased_counts = pd.DataFrame(columns=['department', 'role', 'rank', 'observed_avg'])
        
        comparison = gt_counts.merge(biased_counts, on=['department', 'role', 'rank'], how='left')
        comparison['observed_avg'] = comparison['observed_avg'].fillna(0)
        comparison['observation_rate'] = comparison['role'].map(self.observation_rates)
        comparison['actual_rate'] = comparison['observed_avg'] / comparison['ground_truth_avg']
        
        return comparison.sort_values(['department', 'rank'])
