import gc
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Optional, Tuple, Union

import dictys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dictys.net import stat
from dictys.utils.numpy import ArrayLike, NDArray
from joblib import Memory
from scipy import stats
from scipy.stats import hypergeom
from tqdm import tqdm

from utils_custom import *

class episode_dynamics:
    """
    Workflow for episodic GRN extraction, filtering, force calculation, and enrichment.
    """
    def __init__(self, dictys_dynamic_object, output_folder, latent_factor_folder, trajectory_range=(1, 3), num_points=40, dist=0.001, sparsity=0.01):
        """
        Initialize the episode_dynamics object.
        Args:
            dictys_dynamic_object: The loaded dictys dynamic network object.
            output_folder: Path to save intermediate and output files.
            trajectory_range: (start, stop) tuple for pseudotime/episode.
            num_points: Number of pseudotime points.
            dist: Distance parameter for smoothing.
            sparsity: Sparsity for binarization.
        """
        self.dictys_dynamic_object = dictys_dynamic_object
        self.output_folder = output_folder
        self.latent_factor_folder = latent_factor_folder
        self.trajectory_range = trajectory_range
        self.num_points = num_points
        self.dist = dist
        self.sparsity = sparsity
        self.lcpm_dcurve = None
        self.dtime = None
        self.episode_beta_dcurve = None
        self.filtered_edges = None
        self.filtered_edges_p001 = None
        self.tf_lcpm_episode = None
        self.force_curves = None
        self.avg_force_df = None
        self.episodic_grn_edges = None
        self.lf_genes = None
        self.lf_in_object = None
        self.episodic_enrichment_df = None

    def compute_expression_curves(self, mode="expression"):
        """
        Compute expression curves for all genes.
        """
        lcpm_dcurve, dtime = compute_expression_regulation_curves(
            self.dictys_dynamic_object,
            start=self.trajectory_range[0],
            stop=self.trajectory_range[1],
            num=self.num_points,
            dist=self.dist,
            mode=mode
        )
        self.lcpm_dcurve = lcpm_dcurve
        self.dtime = dtime
        return lcpm_dcurve, dtime

    def build_episode_grn(self, time_slice=slice(0, 5)):
        """
        Build the episodic GRN (weighted and binarized) for the specified episode/time window.
        """
        pts, fsmooth = self.dictys_dynamic_object.linspace(self.trajectory_range[0], self.trajectory_range[1], self.num_points, self.dist)
        stat1_net = fsmooth(stat.net(self.dictys_dynamic_object))
        stat1_netbin = stat.fbinarize(stat1_net, sparsity=self.sparsity)
        dnet = stat1_net.compute(pts)
        dnetbin = stat1_netbin.compute(pts)
        dnet_episode = dnet[:, :, time_slice]
        dnetbin_episode = dnetbin[:, :, time_slice]
        # Map indices to gene names
        ndict = self.dictys_dynamic_object.ndict
        index_to_gene = {idx: name for name, idx in ndict.items()}
        target_names = [index_to_gene[idx] for idx in range(dnetbin_episode.shape[1])]
        tf_gene_indices = [self.dictys_dynamic_object.nids[0][tf_idx] for tf_idx in range(dnetbin_episode.shape[0])]
        tf_names = [index_to_gene[idx] for idx in tf_gene_indices]
        # Reshape to DataFrame
        import pandas as pd
        index_tuples = [(tf, target) for tf in tf_names for target in target_names]
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['TF', 'Target'])
        n_tfs, n_targets, n_times = dnet_episode.shape
        reshaped_data = dnet_episode.reshape(-1, n_times)
        episode_beta_dcurve = pd.DataFrame(
            reshaped_data,
            index=multi_index,
            columns=[f'time_{i}' for i in range(n_times)]
        )
        episode_beta_dcurve = episode_beta_dcurve[episode_beta_dcurve.sum(axis=1) != 0]
        # Remove TFs with names starting with ZNF and ZBTB
        episode_beta_dcurve = episode_beta_dcurve[~episode_beta_dcurve.index.get_level_values(0).str.startswith('ZNF') & ~episode_beta_dcurve.index.get_level_values(0).str.startswith('ZBTB')]
        self.episode_beta_dcurve = episode_beta_dcurve
        return episode_beta_dcurve

    def filter_edges(self, min_nonzero_timepoints=3, alpha=0.05, min_observations=3, check_direction_invariance=True, n_processes=16, chunk_size=8000, pval_threshold=0.001):
        """
        Filter episodic GRN edges for significance and direction invariance.
        """
        filtered_edges = filter_edges_by_significance_and_direction(
            self.episode_beta_dcurve,
            min_nonzero_timepoints=min_nonzero_timepoints,
            alpha=alpha,
            min_observations=min_observations,
            check_direction_invariance=check_direction_invariance,
            n_processes=n_processes,
            chunk_size=chunk_size,
            save_intermediate=False,
            intermediate_path=self.output_folder
        )
        self.filtered_edges = filtered_edges
        filtered_edges_p001 = filtered_edges[filtered_edges['p_value'] < pval_threshold]
        self.filtered_edges_p001 = filtered_edges_p001
        return filtered_edges_p001

    def compute_tf_expression(self):
        """
        Compute TF expression for the episode (matching time window).
        """
        tf_names = self.filtered_edges_p001.index.get_level_values(0).unique()
        tf_lcpm_values = self.lcpm_dcurve.loc[tf_names]
        n_time_cols = len([col for col in self.filtered_edges_p001.columns if col.startswith('time_')])
        tf_lcpm_episode = tf_lcpm_values.iloc[:, 0:n_time_cols]
        tf_lcpm_episode.columns = [col for col in self.filtered_edges_p001.columns if col.startswith('time_')][:n_time_cols]
        self.tf_lcpm_episode = tf_lcpm_episode
        return tf_lcpm_episode

    def calculate_forces(self, n_processes=20, chunk_size=30000, epsilon=1e-10):
        """
        Calculate force curves for the filtered episodic GRN.
        """
        beta_curves_for_force = self.filtered_edges_p001.drop('p_value', axis=1)
        force_curves = calculate_force_curves_parallel(
            beta_curves=beta_curves_for_force,
            tf_expression=self.tf_lcpm_episode,
            n_processes=n_processes,
            chunk_size=chunk_size,
            epsilon=epsilon,
            save_intermediate=False
        )
        self.force_curves = force_curves
        avg_force = force_curves.mean(axis=1)
        avg_force_df = avg_force.to_frame(name='avg_force')
        self.avg_force_df = avg_force_df
        return avg_force_df

    def select_top_edges(self, percentile=99):
        """
        Select the top k% of edges by absolute average force to build the episodic GRN.
        """
        threshold = np.percentile(np.abs(self.avg_force_df['avg_force']), percentile)
        top_percent_mask = np.abs(self.avg_force_df['avg_force']) >= threshold
        episodic_grn_edges = self.avg_force_df[top_percent_mask].copy()
        self.episodic_grn_edges = episodic_grn_edges
        return episodic_grn_edges
    
    def select_top_activating_and_repressing_edges(self, percentile_positive=98.5, percentile_negative=0.5):
        """
        Select the top k% of edges by average force to build the episodic GRN.
        Selects top k% positive and top k% negative edges separately.
        Returns the selected edges as a DataFrame.
        """
        avg_force = self.avg_force_df['avg_force']
        # Separate positive and negative selection
        positive_forces = avg_force[avg_force > 0]
        negative_forces = avg_force[avg_force < 0]
        # Top k% positive
        if len(positive_forces) > 0:
            pos_threshold = np.percentile(positive_forces, percentile_positive)
            top_pos_edges = positive_forces[positive_forces >= pos_threshold]
        else:
            top_pos_edges = pd.Series(dtype=avg_force.dtype)
        # Top k% negative (most negative)
        if len(negative_forces) > 0:
            neg_threshold = np.percentile(negative_forces, percentile_negative)
            top_neg_edges = negative_forces[negative_forces <= neg_threshold]
        else:
            top_neg_edges = pd.Series(dtype=avg_force.dtype)
        episodic_grn_edges = pd.concat([top_pos_edges, top_neg_edges]).to_frame(name='avg_force')
        episodic_grn_edges = episodic_grn_edges.sort_values(by='avg_force', ascending=False)
        self.episodic_grn_edges = episodic_grn_edges
        return episodic_grn_edges

    def set_lf_genes(self, lf_genes):
        """
        Set the list of LF genes for enrichment analysis.
        """
        self.lf_genes = lf_genes
        self.lf_in_object = check_if_gene_in_ndict(self.dictys_dynamic_object, lf_genes, return_index=True)
        return self.lf_in_object

    def annotate_lf_in_grn(self):
        """
        Annotate which targets in the episodic GRN are LF genes.
        """
        if self.lf_genes is None:
            raise ValueError("LF genes not set. Use set_lf_genes() first.")
        self.episodic_grn_edges['is_in_lf'] = self.episodic_grn_edges.index.get_level_values(1).isin(self.lf_genes)
        return self.episodic_grn_edges

    def calculate_enrichment(self):
        """
        Calculate TF enrichment for LF genes in the episodic GRN.
        """
        from utils_custom import calculate_tf_episodic_enrichment
        lf_in_episodic_grn = self.episodic_grn_edges[self.episodic_grn_edges['is_in_lf']]
        lf_genes_active_in_episode = lf_in_episodic_grn.index.get_level_values(1).unique()
        target_genes_in_episodic_grn = self.episodic_grn_edges.index.get_level_values(1).unique()
        enrichment_df = calculate_tf_episodic_enrichment(
            self.episodic_grn_edges,
            total_lf_genes=len(lf_genes_active_in_episode),
            total_genes_in_grn=len(target_genes_in_episodic_grn)
        )
        episodic_enrichment_df_sorted = enrichment_df.sort_values(by='enrichment_score', ascending=False)
        # drop rows with 0 enrichment score
        episodic_enrichment_df_sorted = episodic_enrichment_df_sorted[episodic_enrichment_df_sorted['enrichment_score'] != 0]
        self.episodic_enrichment_df = episodic_enrichment_df_sorted
        return episodic_enrichment_df_sorted

def run_episode(
    episode_idx,
    dictys_dynamic_object_path,
    output_folder,
    latent_factor_folder,
    time_slice_start,
    time_slice_end,
    lf_gene_file,
    percentile_positive=98.5,
    percentile_negative=0.5
):
    # Load dictys object inside the process
    dictys_dynamic_object = dictys.net.dynamic_network.from_file(dictys_dynamic_object_path)
    epi = episode_dynamics(
        dictys_dynamic_object=dictys_dynamic_object,
        output_folder=output_folder,
        latent_factor_folder=latent_factor_folder,
        trajectory_range=(1, 3),  # Adjust as needed
        num_points=40,
        dist=0.001,
        sparsity=0.01
    )
    epi.compute_expression_curves()
    lf_genes = pd.read_csv(lf_gene_file, sep='\t')['names'].tolist()
    epi.set_lf_genes(lf_genes)
    epi.build_episode_grn(time_slice=slice(time_slice_start, time_slice_end))
    epi.filter_edges()
    epi.compute_tf_expression()
    epi.calculate_forces()
    epi.select_top_activating_and_repressing_edges(percentile_positive, percentile_negative)
    epi.annotate_lf_in_grn()
    enrichment_df = epi.calculate_enrichment()
    # Save to CSV
    out_path = os.path.join(output_folder, f'enrichment_episode_{episode_idx}.csv')
    enrichment_df.to_csv(out_path, index=False)
    return out_path