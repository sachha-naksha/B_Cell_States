import numpy as np
import pandas as pd
import os
import joblib
import pickle
import math
import ast
from scipy.stats import median_abs_deviation, hypergeom, mannwhitneyu
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

import dictys
from utils_custom import *

def sort_tfs_by_gene_similarity(tf_genes_dict, method='jaccard_hierarchical', return_linkage=False):
    """
    Sorts TFs based on gene similarity using Jaccard similarity and hierarchical clustering.
    
    Parameters:
    -----------
    tf_genes_dict : dict
        Dictionary mapping TF names to sets of target genes.
    method : str, default 'jaccard_hierarchical'
        Method for sorting. Currently supports 'jaccard_hierarchical'.
    return_linkage : bool, default False
        Whether to return the linkage matrix along with sorted TFs.
        
    Returns:
    --------
    list or tuple
        If return_linkage=False: List of TF names sorted by gene similarity.
        If return_linkage=True: Tuple of (sorted_tfs, linkage_matrix, tf_labels)
    """
    
    all_tfs = list(tf_genes_dict.keys())
    
    if len(all_tfs) <= 1:
        sorted_tfs = sorted(all_tfs)
        if return_linkage:
            return sorted_tfs, None, all_tfs
        return sorted_tfs
    
    def jaccard_similarity(set1, set2):
        """Calculate Jaccard similarity between two gene sets."""
        if len(set1) == 0 and len(set2) == 0:
            return 1.0  # Both empty, consider similar
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    # Create similarity matrix
    n_tfs = len(all_tfs)
    similarity_matrix = np.zeros((n_tfs, n_tfs))
    
    for i, tf1 in enumerate(all_tfs):
        for j, tf2 in enumerate(all_tfs):
            similarity_matrix[i, j] = jaccard_similarity(tf_genes_dict[tf1], tf_genes_dict[tf2])
    
    # Convert to distance matrix (1 - similarity) for clustering
    distance_matrix = 1 - similarity_matrix
    
    try:
        # Convert to condensed distance matrix for linkage
        from scipy.spatial.distance import squareform
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='ward')
        
        # Get the order of TFs from clustering
        cluster_order = leaves_list(linkage_matrix)
        sorted_tfs = [all_tfs[i] for i in cluster_order]
        
        if return_linkage:
            return sorted_tfs, linkage_matrix, all_tfs
        
    except Exception as e:
        # Fallback to alphabetical sorting if clustering fails
        print(f"Warning: Gene similarity clustering failed ({e}), using alphabetical sorting")
        sorted_tfs = sorted(all_tfs)
        if return_linkage:
            return sorted_tfs, None, all_tfs
    
    return sorted_tfs 

def plot_tf_episodic_enrichment_dotplot(
    dfs,
    episode_labels,
    figsize=(12, 8),
    min_dot_size=10,
    max_dot_size=300,
    p_value_threshold=0.05,
    min_significance_threshold=None,
    min_targets_in_lf=2,
    min_targets_dwnstrm=2,
    cmap_name="coolwarm",
    value_legend_title="ES",
    size_legend_title="P-val",
    sort_by_gene_similarity=False,
    show_dendrogram=False,
    dendrogram_ratio=0.2,
    figure_title=None,
    log_scale=False,
    show_plot=True,
    tf_order=None
):
    """
    Plots a dotplot for TF episodic enrichment where dot color represents enrichment score 
    and dot size represents p-value significance (smaller p-value = larger dot).
    
    TFs can be sorted in three ways (priority order):
    1. Custom order (if tf_order is provided)
    2. Gene similarity clustering (if sort_by_gene_similarity=True)
    3. Alphabetical (default)
    
    Parameters:
    -----------
    tf_order : list of str, optional
        Custom order for TF rows. If provided, overrides other sorting methods.
        TFs not in the list will be appended alphabetically at the end.
        TFs in the list but not in the data will be ignored.
    """    
    # 1. Input validation
    required_cols = ['TF', 'p_value', 'enrichment_score', 'genes_in_lf', 'genes_dwnstrm']
    
    for i, df in enumerate(dfs):
        if df is None or df.empty:
            print(f"Episode {i+1} dataframe is None or empty.")
            return None, None, None  # FIXED: Changed from return None, None
        for col in required_cols:
            if col not in df.columns:
                print(f"Episode {i+1} dataframe missing required column: {col}")
                return None, None, None  # FIXED: Changed from return None, None
    
    # 2. Helper function to parse genes_in_lf column
    def parse_genes_in_lf(genes_str):
        """Parse the string representation of genes tuple."""
        try:
            if pd.isna(genes_str) or genes_str == '' or genes_str == '()':
                return set()
            # Handle string representation of tuples
            genes_tuple = ast.literal_eval(genes_str)
            if isinstance(genes_tuple, tuple):
                return set(genes_tuple)
            elif isinstance(genes_tuple, str):
                return {genes_tuple}
            else:
                return set()
        except:
            return set()
                
    # 3. Collect all TFs and their associated genes across all episodes
    tf_genes_dict = {}  # TF -> set of genes across all episodes
    tf_dwnstrm_genes_dict = {}  # TF -> set of downstream genes across all episodes
    plot_data_list = []
    
    for i, (df, episode_label) in enumerate(zip(dfs, episode_labels)):
        df_clean = df.dropna(subset=['TF', 'p_value', 'enrichment_score'])
        
        for _, row in df_clean.iterrows():
            tf_name = row['TF']
            genes_in_lf_set = parse_genes_in_lf(row.get('genes_in_lf', ''))
            genes_dwnstrm_set = parse_genes_in_lf(row.get('genes_dwnstrm', ''))
            
            # Accumulate genes for each TF across episodes
            if tf_name not in tf_genes_dict:
                tf_genes_dict[tf_name] = set()
                tf_dwnstrm_genes_dict[tf_name] = set()
            tf_genes_dict[tf_name].update(genes_in_lf_set)
            tf_dwnstrm_genes_dict[tf_name].update(genes_dwnstrm_set)
            
            plot_data_list.append({
                'episode': episode_label,
                'episode_idx': i,
                'TF': tf_name,
                'p_value': row['p_value'],
                'enrichment_score': row['enrichment_score']
            })
    
    if not plot_data_list:
        print("No valid data found across all episodes.")
        return None, None, None  # FIXED, None  # FIXED: Changed from return None, None
    
    plot_data_df = pd.DataFrame(plot_data_list)
    
    # 4. Filter TFs based on gene count criteria (AND condition)
    valid_tfs = set()
    for tf_name in tf_genes_dict.keys():
        lf_gene_count = len(tf_genes_dict[tf_name])
        dwnstrm_gene_count = len(tf_dwnstrm_genes_dict[tf_name])
        
        if lf_gene_count >= min_targets_in_lf and dwnstrm_gene_count >= min_targets_dwnstrm:
            valid_tfs.add(tf_name)
    
    if not valid_tfs:
        print(f"No TFs meet the criteria: >= {min_targets_in_lf} LF genes AND >= {min_targets_dwnstrm} downstream genes")
        return None, None, None  # FIXED, None  # FIXED: Changed from return None, None
    
    # Filter plot data and gene dictionaries to only include valid TFs
    plot_data_df = plot_data_df[plot_data_df['TF'].isin(valid_tfs)]
    tf_genes_dict = {tf: genes for tf, genes in tf_genes_dict.items() if tf in valid_tfs}
    tf_dwnstrm_genes_dict = {tf: genes for tf, genes in tf_dwnstrm_genes_dict.items() if tf in valid_tfs}
    
    print(f"Filtered to {len(valid_tfs)} TFs that meet gene count criteria")
    
    # 5. Filter TFs based on significance threshold
    if min_significance_threshold is not None:
        # Find TFs that have at least one episode with p-value < min_significance_threshold
        significant_tfs = set()
        tf_min_pvalues = plot_data_df.groupby('TF')['p_value'].min()
        significant_tfs = set(tf_min_pvalues[tf_min_pvalues < min_significance_threshold].index)
        
        if not significant_tfs:
            print(f"No TFs meet the minimum significance threshold of {min_significance_threshold}")
            return None, None, None  # FIXED, None  # FIXED: Changed from return None, None

        # Filter the plot data to only include significant TFs
        plot_data_df = plot_data_df[plot_data_df['TF'].isin(significant_tfs)]
        
        # Update tf_genes_dict to only include significant TFs
        tf_genes_dict = {tf: genes for tf, genes in tf_genes_dict.items() if tf in significant_tfs}
        
        print(f"Further filtered to {len(significant_tfs)} TFs that meet significance threshold < {min_significance_threshold}")

    # 6. Sort TFs based on specified method
    # Priority: custom order > gene similarity > alphabetical
    if tf_order is not None:
        # Use custom order
        available_tfs = set(tf_genes_dict.keys())
        
        # Filter custom order to only include TFs present in the data
        all_tfs_sorted = [tf for tf in tf_order if tf in available_tfs]
        
        # Add any TFs not in custom order (alphabetically sorted)
        remaining_tfs = sorted(available_tfs - set(all_tfs_sorted))
        all_tfs_sorted.extend(remaining_tfs)
        
        linkage_matrix = None
        original_tf_labels = None
        
        if remaining_tfs:
            print(f"Note: {len(remaining_tfs)} TFs not in custom order were added alphabetically: {remaining_tfs}")
            
    elif sort_by_gene_similarity:
        if show_dendrogram:
            all_tfs_sorted, linkage_matrix, original_tf_labels = sort_tfs_by_gene_similarity(
                tf_genes_dict, return_linkage=True)
        else:
            all_tfs_sorted = sort_tfs_by_gene_similarity(tf_genes_dict)
            linkage_matrix = None
            original_tf_labels = None
    else:
        all_tfs_sorted = sorted(tf_genes_dict.keys())
        linkage_matrix = None
        original_tf_labels = None
    
    # 7. Map p-values to dot sizes (updated using Scanpy-like logic)
    def p_value_to_size(p_val):
        if p_val > p_value_threshold:
            return min_dot_size * 0.5  # Smaller dot for non-significant
        # Scale significant p-values
        min_p_cap = 1e-6  # Prevents -log10(0)
        log_p = -np.log10(max(p_val, min_p_cap))
        log_thresh = -np.log10(p_value_threshold)
        log_min_cap = -np.log10(min_p_cap)
        
        if log_min_cap == log_thresh:
            scaled_val = 1.0  # Avoid div-by-zero
        else:
            scaled_val = (log_p - log_thresh) / (log_min_cap - log_thresh)
        
        size = min_dot_size + (max_dot_size - min_dot_size) * min(scaled_val, 1.0)
        return size

    plot_data_df['dot_size'] = plot_data_df['p_value'].apply(p_value_to_size)

    # apply log scale to enrichment score
    if log_scale:
        # add offset
        min_enrichment = plot_data_df['enrichment_score'].min()
        if min_enrichment <= 0:
            offset = abs(min_enrichment) + 1e-6
            plot_data_df['enrichment_score_log'] = np.log2(plot_data_df['enrichment_score'] + offset)
            value_legend_title = f"log2({value_legend_title} + {offset:.1e})"
        else:
            plot_data_df['enrichment_score_log'] = np.log2(plot_data_df['enrichment_score'])
            value_legend_title = f"log2({value_legend_title})"
        
        # Use log-transformed values for coloring
        color_values = plot_data_df['enrichment_score_log']
    else:
        color_values = plot_data_df['enrichment_score']

    # 8. Create coordinate mappings
    episode_x_coords = {label: i for i, label in enumerate(episode_labels)}
    tf_y_coords = {tf: i for i, tf in enumerate(all_tfs_sorted)}
    
    # 9. Create figure with subplots
    if show_dendrogram and linkage_matrix is not None and sort_by_gene_similarity:
        # Create figure with dendrogram and main plot
        fig = plt.figure(figsize=figsize)
        
        # Calculate subplot widths
        dendro_width = dendrogram_ratio
        main_width = 1 - dendrogram_ratio - 0.15  # Leave space for colorbar
        
        # Create subplots
        ax_dendro = fig.add_subplot(1, 2, 1)
        ax_main = fig.add_subplot(1, 2, 2)
        
        # Adjust subplot positions
        dendro_left = 0.05
        main_left = dendro_left + dendro_width + 0.02
        
        ax_dendro.set_position([dendro_left, 0.1, dendro_width, 0.8])
        ax_main.set_position([main_left, 0.1, main_width, 0.8])

        # Plot dendrogram
        dendro_plot = dendrogram(
            linkage_matrix, 
            ax=ax_dendro,
            orientation='left',
            labels=original_tf_labels,
            leaf_font_size=8,
            color_threshold=0.7*max(linkage_matrix[:,2])
        )
        ax_dendro.invert_yaxis()
        ax_dendro.set_ylabel("TF Clustering")
        ax_dendro.set_xlabel("Distance")
        ax_dendro.spines['top'].set_visible(False)
        ax_dendro.spines['right'].set_visible(False)
        
        # Remove x-axis labels for cleaner look
        ax_dendro.tick_params(axis='y', which='both', left=False, labelleft=False)
        
    else:
        # Standard single plot
        fig, ax_main = plt.subplots(figsize=figsize)
    
    # 10. Create scatter plot
    scatter = ax_main.scatter(
        x=plot_data_df['episode'].map(episode_x_coords),
        y=plot_data_df['TF'].map(tf_y_coords),
        s=plot_data_df['dot_size'],
        c=color_values,
        cmap=cmap_name,
        edgecolors='gray',
        linewidths=0.5,
        alpha=0.8
    )
    
    # 11. Axis formatting
    # X-axis (Episodes)
    ax_main.set_xticks(list(episode_x_coords.values()))
    ax_main.set_xticklabels(episode_labels, rotation=0, ha="center")
    # Add horizontal padding to main plot
    x_pad = 0.5  # tweak this to get desired spacing
    ax_main.set_xlim(-x_pad, len(episode_labels) - 1 + x_pad)

    ax_main.set_xlabel("Episodes", fontsize=12, fontweight='bold', labelpad=15)
    
    # Y-axis (TFs)
    ax_main.set_yticks(list(tf_y_coords.values()))
    ax_main.set_yticklabels(all_tfs_sorted)
    ax_main.set_ylabel("TFs", fontsize=12, fontweight='bold')
    # 12. Size legend for P-values
    # Create representative p-values for the legend
    legend_p_values = [0.001, 0.01]
    legend_dots = []
    
    for p_val in legend_p_values:
        size_val = p_value_to_size(p_val)
        if p_val > p_value_threshold:
            label_text = f"{p_value_threshold}"
        else:
            label_text = f"{p_val}"
        legend_dots.append(plt.scatter([], [], s=size_val, c='gray', label=label_text))
    
    # Position the size legend
    if show_dendrogram and linkage_matrix is not None and sort_by_gene_similarity:
        bbox_anchor = (1.25, 0.6)
    else:
        bbox_anchor = (1.18, 0.6)
        
    size_leg = ax_main.legend(
        handles=legend_dots, 
        title=size_legend_title,
        bbox_to_anchor=bbox_anchor, 
        loc='center left',
        labelspacing=1.5, 
        borderpad=1, 
        frameon=True,
        handletextpad=1.5,
        scatterpoints=1
    )
    
    # 13. Horizontal Colorbar for Enrichment Score (positioned below p-value legend)
    if show_dendrogram and linkage_matrix is not None and sort_by_gene_similarity:
        # Create axes for horizontal colorbar below the p-value legend
        cbar_ax = fig.add_axes([0.85, 0.15, 0.3, 0.03])  # [left, bottom, width, height]
    else:
        # Create axes for horizontal colorbar below the p-value legend
        cbar_ax = fig.add_axes([0.65, 0.4, 0.2, 0.02])  # [left, bottom, width, height]
    
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(value_legend_title, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # 14. Final formatting
    sorting_method = "gene similarity" if sort_by_gene_similarity else "alphabetical" 
    ax_main.grid(True, linestyle='--', alpha=0.3, axis='both')
    ax_main.tick_params(axis='both', which='major', pad=5)
    
    # Invert y-axis so first TF is at top
    ax_main.invert_yaxis()
    
    # Add figure title if provided
    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=14, fontweight='bold', y=0.95)

    if show_plot:
        plt.tight_layout()
        plt.show()
    
    return fig, plot_data_df, all_tfs_sorted