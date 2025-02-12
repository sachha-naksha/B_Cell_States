import os
from typing import Optional, Tuple, Union
from joblib import Memory

import dictys
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dictys.net import stat
from dictys.plot.dynamic import _compute_chars_
import ipdb

PATH_TO_CELL_LABELS = "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2/data/clusters.csv"
memory = Memory("./cache", verbose=0)

# To-Do: Create a retrieval class with all functions to retrieve indices, names, weights, etc.
@memory.cache
def load_dynamic_object(dynamic_object_path):
    """
    Load the dynamic object from the given path
    """
    dynamic_object = dictys.net.dynamic_network.from_file(dynamic_object_path)
    return dynamic_object

######################### Indices retrieval #########################
def get_tf_indices(dictys_dynamic_object, tf_list):
    """
    Get the indices of transcription factors from a list, if present in ndict and nids[0].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    tf_mappings_to_gene_hashmap = dictys_dynamic_object.nids[0]
    tf_indices = []
    tf_gene_indices = []
    missing_tfs = []
    for gene in tf_list:
        # Check if the gene is in the gene_hashmap
        if gene in gene_hashmap:
            gene_index = gene_hashmap[gene]  # Get the index in gene_hashmap
            # Check if the gene index is present in tf_mappings_to_gene_hashmap
            match = np.where(tf_mappings_to_gene_hashmap == gene_index)[0]
            if match.size > 0:  # If a match is found
                tf_indices.append(int(match[0]))  # Append the position of the match
                tf_gene_indices.append(int(gene_index))  # Also append the gene index
            else:
                missing_tfs.append(gene)  # Gene exists but not as a TF
        else:
            missing_tfs.append(gene)  # Gene not found at all
    return tf_indices, tf_gene_indices, missing_tfs

def get_gene_indices(dictys_dynamic_object, gene_list):
    """
    Get the indices of target genes from a list, if present in ndict and nids[1].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    gene_indices = []
    for gene in gene_list:
        if gene in gene_hashmap:
            gene_indices.append(gene_hashmap[gene])
    return gene_indices

def get_all_gene_names(dictys_dynamic_object):
    """
    Get all possible target gene names from the dynamic object's ndict
    """
    # Get all gene names from ndict
    gene_names = list(dictys_dynamic_object.ndict.keys())
    gene_names.sort()  # Sort for consistency
    return gene_names

def get_pseudotime_of_windows(dictys_dynamic_object, window_indices):
    """
    Get the pseudotime of specific windows for x-axis in plots
    """
    pseudotime_relative_to_bifurcation = dictys_dynamic_object.point[
        "s"
    ].locs  # Access via dictionary keys
    branch_pseudotime = [
        float(pseudotime_relative_to_bifurcation[idx]) for idx in window_indices
    ]
    return branch_pseudotime

######################### Edge weights retrieval #########################

def get_grn_weights_for_tfs(dictys_dynamic_object, tf_indices=None, window_indices=None, nonzero_fraction_threshold=0.1):
    """
    Get the weights matrix for specified TFs over specific windows
    """
    # Get the weights array
    global_grn_weights = dictys_dynamic_object.prop["es"]["w_n"]
    # Convert indices to numpy arrays and ensure they're integers
    tf_indices = np.array(tf_indices, dtype=int) if tf_indices is not None else np.array([], dtype=int)
    window_indices = np.array(window_indices, dtype=int) if window_indices is not None else np.array([], dtype=int)
    # Get dimensions
    n_targets = global_grn_weights.shape[1]
    # Calculate non-zero fractions across time for the specified TFs
    nonzero_edge_fraction = np.mean(global_grn_weights[tf_indices] != 0, axis=2)  # shape: (n_selected_tfs, n_targets)
    # Create sparsity mask
    sparsity_mask = nonzero_edge_fraction >= nonzero_fraction_threshold  # shape: (n_selected_tfs, n_targets)
    # Extract weights for specified TFs and windows
    weights = global_grn_weights[tf_indices, :, :]  # shape: (n_selected_tfs, n_targets, n_windows)
    weights = weights[:, :, window_indices]  # Select specific windows
    # Apply sparsity mask
    weights = np.where(sparsity_mask[:, :, np.newaxis], weights, 0)
    print(f"Output shape: {weights.shape} (n_tfs={len(tf_indices)}, n_targets={n_targets}, n_windows={len(window_indices)})")
    return weights

def get_grn_weights_for_windows(dictys_dynamic_object, window_indices=None, nonzero_fraction_threshold=0.1):
    """
    Get the weights matrix for specified TFs over specific windows
    """
    # Get the weights array
    global_grn_weights = dictys_dynamic_object.prop["es"]["w_n"]
    # Convert indices to numpy arrays and ensure they're integers
    window_indices = np.array(window_indices, dtype=int) if window_indices is not None else np.array([], dtype=int)
    # Calculate non-zero fractions across time for the specified TFs
    nonzero_edge_fraction = np.mean(global_grn_weights != 0, axis=2)  # shape: (n_selected_tfs, n_targets)
    # Create sparsity mask
    sparsity_mask = nonzero_edge_fraction >= nonzero_fraction_threshold  # shape: (n_selected_tfs, n_targets)
    # Extract weights for specified windows
    weights = global_grn_weights[:, :, window_indices]
    # Apply sparsity mask
    weights = np.where(sparsity_mask[:, :, np.newaxis], weights, 0)
    print(f"Output shape: {weights.shape} n_windows={len(window_indices)})")
    return weights

def get_indirect_grn_weights_for_tfs(
    dictys_dynamic_object, tf_indices=None, window_indices=None, nonzero_fraction_threshold=0.1
):
    """
    Get the indirect weights of TFs over specific windows for x-axis in plots
    """
    # Get the weights array
    global_grn_weights = dictys_dynamic_object.prop["es"]["w_in"]
    # Apply sparsity filter to global GRN first
    nonzero_edge_fraction_across_pseudotime = (global_grn_weights != 0).mean(axis=2)  # mean of a boolean, shape is total TFs x total targets
    edge_sparsity_filter = nonzero_edge_fraction_across_pseudotime >= nonzero_fraction_threshold
    # Convert indices to lists if they're not already
    tf_indices = list(tf_indices) if tf_indices is not None else []
    window_indices = list(window_indices) if window_indices is not None else []
    # Get the indirect weights of the specified TFs over the specified windows
    indirect_weights_of_tf_target = global_grn_weights[np.ix_(tf_indices, slice(None), window_indices)]
    # Apply sparsity mask to final weights (in case any edges don't meet threshold)
    final_sparsity_mask = edge_sparsity_filter[np.ix_(tf_indices, slice(None))]
    indirect_weights_of_tf_target = np.where(final_sparsity_mask[:, :, np.newaxis], indirect_weights_of_tf_target, 0)
    return indirect_weights_of_tf_target

def get_grn_weights_for_tf_target_pairs(
    dictys_dynamic_object, tf_indices=None, target_indices=None, window_indices=None, nonzero_fraction_threshold=0.1
):
    """
    Get the weights for specific TF-target pairs over specific windows.
    """
    # Get the weights array
    global_grn_weights = dictys_dynamic_object.prop["es"]["w_n"]
    # Apply sparsity filter to global GRN first
    nonzero_edge_fraction_across_pseudotime = (global_grn_weights != 0).mean(axis=2)  # mean of a boolean, shape is total TFs x total targets
    edge_sparsity_filter = nonzero_edge_fraction_across_pseudotime >= nonzero_fraction_threshold
    # Get weights of the specified TF-target pairs over the specified windows
    weights_of_tf_target = global_grn_weights[np.ix_(tf_indices, target_indices, window_indices)]
    # Apply sparsity mask to final weights (in case any edges don't meet threshold)
    final_sparsity_mask = edge_sparsity_filter[np.ix_(tf_indices, target_indices)]
    weights_of_tf_target = np.where(final_sparsity_mask[:, :, np.newaxis], weights_of_tf_target, 0)
    return weights_of_tf_target

def get_one_hop_grn_weights(dictys_dynamic_object, tf_indices=None, gene_indices=None, window_indices=None, nonzero_fraction_threshold=0.1):
    """
    Get the LF intersection with GRN weight matrix. Add new TF nodes that regulate the specified target genes. Don't add new target nodes.
    """
    # Get the weights array
    global_grn_weights = dictys_dynamic_object.prop["es"]["w_n"]
    # Apply sparsity filter to global GRN first (across all windows, there can be sparsity in one branch but not the other)
    nonzero_edge_fraction_across_all_windows = (global_grn_weights != 0).mean(axis=2)  # mean of a boolean, shape is total TFs x total targets
    edge_sparsity_filter = nonzero_edge_fraction_across_all_windows >= nonzero_fraction_threshold
    # Convert query indices to lists if they're not already
    tf_indices = list(tf_indices) if tf_indices is not None else []
    gene_indices = list(gene_indices) if gene_indices is not None else []
    # use all windows if indices are not specified
    window_indices = list(range(global_grn_weights.shape[2])) if window_indices is None else window_indices
    # Augment the TF nodes that regulate the specified target genes (considering sparsity)
    if gene_indices:
        # Only consider edges that pass sparsity filter
        tf_mask = np.any(edge_sparsity_filter[:, gene_indices], axis=1)
        regulating_tf_indices = np.where(tf_mask)[0]
        # Combine with specified TF indices, removing duplicates while preserving order
        all_tf_indices = list(dict.fromkeys(tf_indices + list(regulating_tf_indices)))
    else:
        all_tf_indices = tf_indices
    
    # Extract the combined weights matrix using the computed indices
    combined_weights = global_grn_weights[np.ix_(all_tf_indices, gene_indices, window_indices)]
    # Apply sparsity mask to final weights (in case non-augmented edges don't meet threshold)
    final_sparsity_mask = edge_sparsity_filter[np.ix_(all_tf_indices, gene_indices)]
    combined_weights = np.where(final_sparsity_mask[:, :, np.newaxis], combined_weights, 0)
    return combined_weights, final_sparsity_mask, all_tf_indices

##################################### Similarity metrics ############################################
def get_sign_switching_pairs(pb_weights, gc_weights, tf_names, target_names, 
                           edge_presence_threshold=0.3,
                           mean_weight=0.3,
                           spread_weight=0.2,
                           min_sign_changes=2):
    """
    Find TF-target pairs that show regulation sign changes between PB and GC branches,
    prioritizing by number of sign changes and considering mean and spread differences.
    """
    # Verify dimensions
    assert len(tf_names) == pb_weights.shape[0], "Number of TF names doesn't match weight matrix"
    assert len(target_names) == pb_weights.shape[1], "Number of target names doesn't match weight matrix"
    n_windows = min(pb_weights.shape[2], gc_weights.shape[2])
    # Calculate edge presence in each branch
    pb_edge_presence = np.mean(pb_weights[:, :, :n_windows] != 0, axis=2)
    gc_edge_presence = np.mean(gc_weights[:, :, :n_windows] != 0, axis=2)
    edge_present = (pb_edge_presence > edge_presence_threshold) & \
                  (gc_edge_presence > edge_presence_threshold)
    # Initialize arrays to store metrics
    sign_changes = np.zeros((pb_weights.shape[0], pb_weights.shape[1]))
    mean_diffs = np.zeros_like(sign_changes)
    spread_diffs = np.zeros_like(sign_changes)
    # Calculate metrics for each TF-target pair
    for i in range(pb_weights.shape[0]):
        for j in range(pb_weights.shape[1]):
            if edge_present[i, j]:
                pb_vals = pb_weights[i, j, :n_windows]
                gc_vals = gc_weights[i, j, :n_windows]
                # Count sign changes
                sign_diff = np.sign(pb_vals) != np.sign(gc_vals)
                sign_changes[i, j] = np.sum(sign_diff & (pb_vals != 0) & (gc_vals != 0))
                # Calculate mean difference
                pb_mean = np.mean(pb_vals)
                gc_mean = np.mean(gc_vals)
                mean_diffs[i, j] = np.abs(pb_mean - gc_mean)
                # Calculate spread difference (using standard deviation)
                pb_spread = np.std(pb_vals)
                gc_spread = np.std(gc_vals)
                spread_diffs[i, j] = np.abs(pb_spread - gc_spread)
    # Normalize mean and spread differences to [0, 1] range
    mean_diffs_norm = mean_diffs / np.max(mean_diffs) if np.max(mean_diffs) > 0 else mean_diffs
    spread_diffs_norm = spread_diffs / np.max(spread_diffs) if np.max(spread_diffs) > 0 else spread_diffs
    # Filter pairs with more than minimum sign changes
    sufficient_changes = sign_changes >= min_sign_changes
    # Calculate composite scores
    # Note: sign_changes are not normalized as they are the primary filter
    composite_scores = (sign_changes + 
                       mean_weight * mean_diffs_norm + 
                       spread_weight * spread_diffs_norm)
    # Get indices of pairs that pass filters
    tf_idx, target_idx = np.where(sufficient_changes & edge_present)
    # Get scores and sort
    scores = composite_scores[tf_idx, target_idx]
    sort_idx = np.argsort(-scores)
    # Create output lists
    filtered_pairs = []
    pair_scores = []
    pair_metrics = []
    for i in sort_idx:
        tf_i = tf_idx[i]
        target_i = target_idx[i]
        filtered_pairs.append((tf_names[tf_i], target_names[target_i]))
        pair_scores.append(scores[i])
        # Store detailed metrics
        pair_metrics.append({
            'sign_changes': sign_changes[tf_i, target_i],
            'mean_diff': mean_diffs[tf_i, target_i],
            'spread_diff': spread_diffs[tf_i, target_i],
            'pb_pattern': pb_weights[tf_i, target_i, :n_windows],
            'gc_pattern': gc_weights[tf_i, target_i, :n_windows]
        })
    return filtered_pairs, pair_scores, pair_metrics

# import numpy as np
# from scipy import sparse
# from scipy.spatial.distance import cosine
# from scipy.fft import fft
# import pandas as pd

# def compute_signal_similarity(signal1, signal2, method='fft'):
#     """
#     Compute similarity between two time series signals.
#     """
#     if method == 'fft':
#         # Compute FFT and get magnitudes
#         fft1 = np.abs(fft(signal1))
#         fft2 = np.abs(fft(signal2))
#         # Compute cosine similarity between FFT magnitudes
#         similarity = 1 - cosine(fft1, fft2)
#     elif method == 'cosine':
#         similarity = 1 - cosine(signal1, signal2)
#     elif method == 'pearson':
#         similarity = np.corrcoef(signal1, signal2)[0, 1]
#     return similarity

# def create_edge_similarity_matrix(gc_weights, pb_weights, method='fft'):
#     """
#     Create sparse matrix of edge similarities between GC and PB branches.
#     """
#     n_tfs, n_targets, _ = gc_weights.shape
#     # Lists to store COO format data
#     rows = []
#     cols = []
#     data = []
#     for tf_idx in range(n_tfs):
#         for target_idx in range(n_targets):
#             gc_signal = gc_weights[tf_idx, target_idx, :]
#             pb_signal = pb_weights[tf_idx, target_idx, :]
#             # Only compute similarity if either signal has non-zero variance
#             if np.var(gc_signal) > 0 or np.var(pb_signal) > 0:
#                 similarity = compute_signal_similarity(gc_signal, pb_signal, method)
#                 # Store non-zero similarities
#                 if not np.isnan(similarity) and abs(similarity) > 1e-6:
#                     rows.append(tf_idx)
#                     cols.append(target_idx)
#                     data.append(similarity)
#     # Create sparse matrix
#     similarity_matrix = sparse.coo_matrix(
#         (data, (rows, cols)), 
#         shape=(n_tfs, n_targets)
#     )
#     return similarity_matrix

# def get_divergent_edges(similarity_matrix, dictys_dynamic_object, n_top=10):
#     """
#     Get most divergent TF-target edges based on similarity scores and map indices back to gene names.
#     """
#     # Create reverse mapping once
#     idx_to_gene = {idx: gene for gene, idx in dictys_dynamic_object.ndict.items()}
#     # Convert to dense for easier manipulation
#     dense_matrix = similarity_matrix.todense()
#     # Get ONLY top n_top flat indices
#     flat_matrix = np.asarray(dense_matrix).ravel()
#     flat_indices = np.argsort(flat_matrix)[:n_top]  # Get only n_top indices
#     # Convert flat indices to 2D indices
#     tf_indices, target_indices = np.unravel_index(flat_indices, dense_matrix.shape)
#     # Map indices back to gene names
#     tf_names = []
#     target_names = []
#     scores = []
#     for tf_idx, target_idx in zip(tf_indices, target_indices):
#         # Get TF gene index and name
#         tf_gene_idx = dictys_dynamic_object.nids[0][tf_idx].item()
#         tf_name = idx_to_gene[tf_gene_idx]
#         tf_names.append(tf_name)
#         # Get target name
#         target_name = idx_to_gene[target_idx]
#         target_names.append(target_name)
#         # Get similarity score
#         score = float(dense_matrix[tf_idx, target_idx])
#         scores.append(score)
#     # Create DataFrame of results
#     results = pd.DataFrame({
#         'TF_idx': tf_indices,
#         'Target_idx': target_indices,
#         'TF_name': tf_names,
#         'Target_name': target_names,
#         'Similarity_score': scores
#     })
#     return results

##################################### Utils ############################################

def get_state_labels_in_window(dictys_dynamic_object, cell_labels):
    """
    Creates a mapping of window indices to their constituent cells' labels
    """
    # get cell assignment matrix from dictys_dynamic_object.prop['sc']['w']
    cell_assignment_matrix = dictys_dynamic_object.prop['sc']['w']
    state_labels_in_window = {}
    for window_idx in range(cell_assignment_matrix.shape[0]):
        indices_of_cells_present_in_window = np.where(cell_assignment_matrix[window_idx] == 1)[0]
        state_labels_in_window[window_idx] = [cell_labels[idx] for idx in indices_of_cells_present_in_window]
    return state_labels_in_window

def get_state_total_counts(cell_labels):
    """
    Get total number of cells for each state in the dataset
    """
    state_counts = {}
    for label in cell_labels:
        state_counts[label] = state_counts.get(label, 0) + 1
    return state_counts

def get_top_k_fraction_labels(dictys_dynamic_object, window_idx, cell_labels, k=2):
    """
    Returns the k labels with both fractions for a given window
    """
    # Get state labels for all windows
    state_labels_dict = get_state_labels_in_window(dictys_dynamic_object, cell_labels)
    # Get window labels for specified window
    window_labels = state_labels_dict[window_idx]
    # Get total counts across all states
    state_total_counts = get_state_total_counts(cell_labels)
    # Count cells per state in this window
    window_counts = {}
    for label in window_labels:
        window_counts[label] = window_counts.get(label, 0) + 1
    total_cells_in_window = len(window_labels)
    # Calculate both fractions for each state
    state_metrics = {}
    for state in window_counts:
        window_composition = window_counts[state] / total_cells_in_window
        state_distribution = window_counts[state] / state_total_counts[state]
        state_metrics[state] = (window_composition, state_distribution)
    # Sort primarily by state_distribution, then by window_composition
    sorted_states = sorted(state_metrics.items(),
                         key=lambda x: (x[1][1], x[1][0]),
                         reverse=True)
    return sorted_states[:k]

def compute_chars(
    dictys_dynamic_object,
    start: int,
    stop: int,
    num: int = 100,
    dist: float = 1.5,
    mode: str = "regulation",
    sparsity: float = 0.01,
) -> pd.DataFrame:
    """
    Compute curve characteristics for one branch.
    """
    pts, fsmooth = dictys_dynamic_object.linspace(start, stop, num, dist)
    if mode == "regulation":
        # Log number of targets
        stat1_net = fsmooth(stat.net(dictys_dynamic_object))
        stat1_netbin = stat.fbinarize(stat1_net, sparsity=sparsity)
        stat1_y = stat.flnneighbor(stat1_netbin)
    elif mode == "expression":
        stat1_y = fsmooth(stat.lcpm(dictys_dynamic_object, cut=0))
    else:
        raise ValueError(f"Unknown mode {mode}.")
    # Pseudo time
    stat1_x = stat.pseudotime(dictys_dynamic_object, pts)
    tmp_y = stat1_y.compute(pts)
    tmp_x = stat1_x.compute(pts)
    dy = pd.DataFrame(tmp_y, index=stat1_y.names[0])
    dx = pd.Series(tmp_x[0]) #first gene's pseudotime is taken as all genes have the same pseudotime
    return dy, dx

def intersect_grn_edges_with_lf(dictys_dynamic_object, lf_genes):
    """
    Get 1-hop GRN edges for latent factor genes; remove sparse edges which show no regulatory activity across majority of cellular transitions
    """
    # 1. Map the gene names in LF to indices (TF and target both)
    lf_tf_indices, lf_tf_gene_indices, _ = get_tf_indices(dictys_dynamic_object, lf_genes)
    lf_gene_indices = get_gene_indices(dictys_dynamic_object, lf_genes)

    # 2. Get the one-hop GRN edges for the LF genes to increase TF nodes. (adjust for sparsity across all windows in this step)
    weights, tf_target_edge_mask, target_one_hop_tf_nodes = get_one_hop_grn_weights(dictys_dynamic_object, lf_tf_indices, lf_gene_indices, window_indices=None, nonzero_fraction_threshold=0.3)
    # 3. Create TF-Target names tuples list of all edges present in the final sparsity mask returned from step 2. (mapping indices back to name)
    # Create reverse mapping once
    idx_to_gene = {idx: gene for gene, idx in dictys_dynamic_object.ndict.items()}
    # get indices of TFs and Targets from mask
    tf_idx = tf_target_edge_mask[0]
    target_gene_idx = tf_target_edge_mask[1]
    # map tf indices to their gene indices
    tf_gene_idx = dictys_dynamic_object.nids[0][tf_idx].item()
    # map indices back to names
    tf_names = [idx_to_gene[tf_gene_idx]]
    target_names = [idx_to_gene[target_gene_idx]]
    # output only the existing edges (pairs) in the sparsity mask
    tf_target_pairs = []
    for i in range(len(tf_names)):
        for j in range(len(target_names)):
            if tf_target_edge_mask[i, j]:  # Check if edge exists in sparsity mask
                tf_target_pairs.append((tf_names[i], target_names[j]))
    return tf_target_pairs

##################################### Plotting ############################################
def fig_regulation_heatmap(
    network: dictys.net.dynamic_network,
    start: int,
    stop: int,
    regulations: list[Tuple[str, str]],
    num: int = 100,
    dist: float = 1.5,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = 'coolwarm',
    figsize: Tuple[float, float] = (2, 0.15),
    vmax: Optional[float] = None
) -> Tuple[matplotlib.pyplot.Figure, matplotlib.axes.Axes, matplotlib.cm.ScalarMappable]:
    """
    Draws pseudo-time dependent heatmap of regulation strengths without clustering.
    Maintains the original order of regulation pairs as provided in the input list.
    """
    # Get dynamic network edge strength
    pts, fsmooth = network.linspace(start, stop, num, dist)
    stat1_net = fsmooth(stat.net(network))
    stat1_x = stat.pseudotime(network, pts)
    tmp = stat1_x.compute(pts)[0]
    dx = pd.Series(tmp)
    # Test regulation existence and extract regulations
    tdict = [dict(zip(x, range(len(x)))) for x in stat1_net.names]
    t1 = [[x[y] for x in regulations if x[y] not in tdict[y]] for y in range(2)]
    if len(t1[0]) > 0:
        raise ValueError("Regulator gene(s) {} not found in network.".format("/".join(t1[0])))
    if len(t1[1]) > 0:
        raise ValueError("Target gene(s) {} not found in network.".format("/".join(t1[1])))
    # Extract regulations to draw
    dnet = stat1_net.compute(pts)
    t1 = np.array([[tdict[0][x[0]], tdict[1][x[1]]] for x in regulations]).T
    dnet = dnet[t1[0], t1[1]]
    # Create figure and axes
    if ax is None:
        figsize = (figsize[0], figsize[1] * dnet.shape[0])
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        if figsize is not None:
            raise ValueError("figsize should not be set if ax is set.")
        fig = ax.get_figure()
        figsize = fig.get_size_inches()
    aspect = (figsize[1] / dnet.shape[0]) / (figsize[0] / dnet.shape[1])
    # Determine and apply colormap
    if isinstance(cmap, str):
        if vmax is None:
            vmax = np.quantile(np.abs(dnet).ravel(), 0.95)
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax), 
            cmap=cmap
        )
    elif vmax is not None:
        raise ValueError("vmax should not be set if cmap is a matplotlib.cm.ScalarMappable.")
    if hasattr(cmap, "to_rgba"):
        im = ax.imshow(cmap.to_rgba(dnet), aspect=aspect, interpolation='none')
    else:
        im = ax.imshow(dnet, aspect=aspect, interpolation='none', cmap=cmap)
        plt.colorbar(im, label="Regulation strength")
    # Set pseudotime labels as x axis labels
    ax.set_xlabel("Pseudotime")
    num_ticks = 10
    tick_positions = np.linspace(0, dnet.shape[1] - 1, num_ticks, dtype=int)
    tick_labels = dx.iloc[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{x:.6f}" for x in tick_labels], rotation=45, ha="right")
    # Set regulation pair labels
    ax.set_yticks(list(range(len(regulations))))
    ax.set_yticklabels(["-".join(x) for x in regulations])
    # Add grid lines to separate rows
    ax.set_yticks(np.arange(dnet.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    return fig, ax, dnet

def cluster_heatmap(d,
		optimal_ordering=True,
		method='ward',
		metric='euclidean',
		dshow=None,
		fig=None,
		cmap='coolwarm',
		aspect=0.1,
		figscale=0.02,
		dtop=0.3,
		dright=0,
		wcolorbar=0.03,
		wedge=0.03,
		xselect=None,
		yselect=None,
		xtick=False,
		ytick=True,
		vmin=None,
		vmax=None,
		inverty=True):
	"""
	Draw 2-D hierachical clustering of pandas.DataFrame, with optional hierachical clustering on both axes.
	X/Y axis of figure corresponds to columns/rows of the dataframe.
	d:		Pandas.DataFrame 2D data with index & column names for clustering.
	optimal_ordering: passed to scipy.cluster.hierarchy.dendrogram
	method: Method of hierarchical clustering, passed to scipy.cluster.hierarchy.linkage.
			Accepts single strs or a tuple of two strings for different method options for x and y.
	metric:	Metric to compute pariwise distance for clustering, passed to scipy.spatial.distance.pdist.
			Accepts single strs or a tuple of two strings for different metric options for x and y.
	dshow:	Pandas.DataFrame 2D data with index & column names to draw heatmap. Defaults to d.
	fig:	Figure to plot on.
	cmap:	Colormap
	aspect:	Aspect ratio
	figscale:	Scale of figure compared to font.
	dtop,
	dright:	Top and right dendrogram size. Value from 0 to 1 values as proportion.
			If 0, do not cluster on given axis.
	wcolorbar: Width of colorbar. Value from 0 to 1 values as proportion.
	wedge:	Width of edges and between colorbar and main figure.
			Value from 0 to 1 values as proportion.
	xselect,
	yselect:np.array(bool) of coordinates to draw. Current only selected coordinates are used for clustering.
			TODO: All coordinates in d are used for clustering.
	xtick,
	ytick:	Whether to show ticks.
	vmin,
	vmax:	Minimum/maximum values of heatmap.
	inverty:Whether to invert direction of y.
	Return:
	figure:	plt.Figure drawn on
	x:		column IDs included
	y:		index IDs included.
	"""
	import matplotlib.pyplot as plt
	from scipy.cluster.hierarchy import dendrogram, linkage
	import numpy as np
	assert isinstance(xtick,bool) or (isinstance(xtick,list) and len(xtick) == d.shape[1])
	assert isinstance(ytick,bool) or (isinstance(ytick,list) and len(ytick) == d.shape[0])
	if isinstance(method,str):
		method=[method,method]
	if len(method)!=2:
		raise ValueError('Parameter "method" must have size 2 for x and y respectively.')
	if isinstance(metric,str):
		metric=[metric,metric]
	if metric is not None and len(metric)!=2:
		raise ValueError('Parameter "metric" must have size 2 for x and y respectively.')
	if metric is None:
		assert d.ndim==2 and d.shape[0]==d.shape[1]
		assert (d.index==d.columns).all()
		assert method[0]==method[1]
		if xselect is not None:
			assert yselect is not None
			assert (xselect==yselect).all()
		else:
			assert yselect is None
	if dshow is None:
		dshow=d
	assert dshow.shape==d.shape and (dshow.index==d.index).all() and (dshow.columns==d.columns).all()
	xt0 = d.columns if isinstance(xtick,bool) else xtick
	yt0 = d.index if isinstance(ytick,bool) else ytick
	# Genes to highlight
	d2 = d.copy()
	if xselect is not None:
		d2 = d2.loc[:, xselect]
		dshow=dshow.loc[:,xselect]
		xt0 = [xt0[x] for x in np.nonzero(xselect)[0]]
	if yselect is not None:
		d2 = d2.loc[yselect]
		dshow=dshow.loc[yselect]
		yt0 = [yt0[x] for x in np.nonzero(yselect)[0]]

	wtop = dtop / (1 + d2.shape[0] / 8)
	wright = dright / (1 + d2.shape[1] * aspect / 8)
	iscolorbar = wcolorbar > 0
	t1 = np.array(d2.T.shape)
	t1 = t1 * figscale
	t1[1] /= aspect
	t1[1] /= 1 - wedge * 2 - wtop
	t1[0] /= 1 - wedge * (2 + iscolorbar) - wright - wcolorbar
	if fig is None:
		fig = plt.figure(figsize=t1)

	d3 = dshow.copy()
	if metric is not None:
		# Right dendrogram
		if dright > 0:
			ax1 = fig.add_axes([
				1 - wedge * (1 + iscolorbar) - wright - wcolorbar, wedge, wright,
				1 - 2 * wedge - wtop])
			tl1 = linkage(d2, method=method[1], metric=metric[1], optimal_ordering=optimal_ordering)
			td1 = dendrogram(tl1, orientation='right')
			ax1.set_xticks([])
			ax1.set_yticks([])
			d3 = d3.iloc[td1['leaves'], :]
			yt0 = [yt0[x] for x in td1['leaves']]
		else:
			ax1=None

		# Top dendrogram
		if dtop > 0:
			ax2 = fig.add_axes([wedge, 1 - wedge - wtop, 1 - wedge * (2 + iscolorbar) - wright - wcolorbar, wtop])
			tl2 = linkage(d2.T, method=method[0], metric=metric[0], optimal_ordering=optimal_ordering)
			td2 = dendrogram(tl2)
			ax2.set_xticks([])
			ax2.set_yticks([])
			d3 = d3.iloc[:, td2['leaves']]
			xt0 = [xt0[x] for x in td2['leaves']]
		else:
			ax2=None
	else:
		if dright > 0 or dtop > 0:
			from scipy.spatial.distance import squareform
			tl1 = linkage(squareform(d2), method=method[0], optimal_ordering=optimal_ordering)
			# Right dendrogram
			if dright > 0:
				ax1 = fig.add_axes([
					1 - wedge * (1 + iscolorbar) - wright - wcolorbar, wedge, wright,
					1 - 2 * wedge - wtop])
				td1 = dendrogram(tl1, orientation='right')
				ax1.set_xticks([])
				ax1.set_yticks([])
			else:
				ax1=None
				td1=None
			# Top dendrogram
			if dtop > 0:
				ax2 = fig.add_axes([wedge, 1 - wedge - wtop, 1 - wedge * (2 + iscolorbar) - wright - wcolorbar, wtop])
				td2 = dendrogram(tl1)
				ax2.set_xticks([])
				ax2.set_yticks([])
			else:
				ax2=None
				td2=None
			td0=td1['leaves'] if td1 is not None else td2['leaves']
			d3 = d3.iloc[td0,:].iloc[:,td0]
			xt0,yt0 = [[y[x] for x in td0] for y in [xt0,yt0]]
	axmatrix = fig.add_axes([
		wedge, wedge, 1 - wedge * (2 + iscolorbar) - wright - wcolorbar,
		1 - 2 * wedge - wtop])
	ka = {'aspect': 1 / aspect, 'origin': 'lower', 'cmap': cmap}
	if vmin is not None:
		ka['vmin'] = vmin
	if vmax is not None:
		ka['vmax'] = vmax
	im = axmatrix.matshow(d3, **ka)
	if not isinstance(xtick,bool) or xtick:
		t1 = list(zip(range(d3.shape[1]), xt0))
		t1 = list(zip(*list(filter(lambda x: x[1] is not None, t1))))
		axmatrix.set_xticks(t1[0])
		axmatrix.set_xticklabels(t1[1], minor=False, rotation=90)
	else:
		axmatrix.set_xticks([])
	if not isinstance(ytick,bool)or ytick:
		t1 = list(zip(range(d3.shape[0]), yt0))
		t1 = list(zip(*list(filter(lambda x: x[1] is not None, t1))))
		axmatrix.set_yticks(t1[0])
		axmatrix.set_yticklabels(t1[1], minor=False)
	else:
		axmatrix.set_yticks([])

	axmatrix.tick_params(top=False,
		bottom=True,
		labeltop=False,
		labelbottom=True,
		left=True,
		labelleft=True,
		right=False,
		labelright=False)
	if inverty:
		if ax1 is not None:
			ax1.set_ylim(ax1.get_ylim()[::-1])
		axmatrix.set_ylim(axmatrix.get_ylim()[::-1])
	if wcolorbar > 0:
		cax = fig.add_axes([
			1 - wedge - wcolorbar, wedge, wcolorbar, 1 - 2 * wedge - wtop])
		fig.colorbar(im, cax=cax)

	return fig, d3.columns, d3.index

def fig_expression_gradient_heatmap(
    network: dictys.net.dynamic_network,
    start: int,
    stop: int,
    genes_or_regulations: Union[list[str], list[Tuple[str, str]]],
    num: int = 100,
    dist: float = 1.5,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    figsize: Tuple[float, float] = (2, 0.15)
) -> Tuple[matplotlib.pyplot.Figure, matplotlib.axes.Axes, matplotlib.cm.ScalarMappable]:
    """
    Draws pseudo-time dependent heatmap of expression gradients.
    """
    # Get expression data
    dy, dx = compute_chars(network, start, stop, num, dist, mode='expression')
    # Determine if input is gene list or regulation list
    if isinstance(genes_or_regulations[0], tuple):
        # Extract target genes from regulations
        target_genes = [target for _, target in genes_or_regulations]
        # Remove duplicates while preserving order
        target_genes = list(dict.fromkeys(target_genes))
    else:
        # Use gene list directly
        target_genes = list(dict.fromkeys(genes_or_regulations))
    # Calculate gradients for target genes
    gradients = np.vstack([
        np.gradient(dy.loc[gene].values, dx.values) 
        for gene in target_genes
    ])
    # Create figure and axes
    if ax is None:
        figsize = (figsize[0], figsize[1] * len(target_genes))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        if figsize is not None:
            raise ValueError("figsize should not be set if ax is set.")
        fig = ax.get_figure()
        figsize = fig.get_size_inches()
    aspect = (figsize[1] / len(target_genes)) / (figsize[0] / gradients.shape[1])
    # Determine and apply colormap
    if isinstance(cmap, str):
        vmax = np.quantile(np.abs(gradients).ravel(), 0.95)
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax), 
            cmap=cmap
        )
    if hasattr(cmap, "to_rgba"):
        im = ax.imshow(cmap.to_rgba(gradients), aspect=aspect, interpolation='none')
    else:
        im = ax.imshow(gradients, aspect=aspect, interpolation='none', cmap=cmap)
        plt.colorbar(im, label="Expression gradient (Δ Log CPM/Δ Pseudotime)")
    # Set pseudotime labels as x axis labels
    ax.set_xlabel("Pseudotime")
    num_ticks = 10
    tick_positions = np.linspace(0, gradients.shape[1] - 1, num_ticks, dtype=int)
    tick_labels = dx.iloc[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{x:.6f}" for x in tick_labels], rotation=45, ha="right")
    # Set target gene labels
    ax.set_yticks(list(range(len(target_genes))))
    ax.set_yticklabels(target_genes)
    # Add grid lines to separate rows
    ax.set_yticks(np.arange(len(target_genes) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    return fig, ax, cmap 

def fig_expression_heatmap(
    network: dictys.net.dynamic_network,
    start: int,
    stop: int,
    genes_or_regulations: Union[list[str], list[Tuple[str, str]]],
    num: int = 100,
    dist: float = 1.5,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    figsize: Tuple[float, float] = (2, 0.15)
) -> Tuple[matplotlib.pyplot.Figure, matplotlib.axes.Axes, matplotlib.cm.ScalarMappable]:
    """
    Draws pseudo-time dependent heatmap of expression values.
    """
    # Get expression data
    dy, dx = compute_chars(network, start, stop, num, dist, mode='expression')
    
    # Determine if input is gene list or regulation list
    if isinstance(genes_or_regulations[0], tuple):
        # Extract target genes from regulations
        target_genes = [target for _, target in genes_or_regulations]
        # Remove duplicates while preserving order
        target_genes = list(dict.fromkeys(target_genes))
    else:
        # Use gene list directly
        target_genes = list(dict.fromkeys(genes_or_regulations))
    
    # Get expression values for target genes (instead of gradients)
    expressions = np.vstack([
        dy.loc[gene].values
        for gene in target_genes
    ])
    
    # Create figure and axes
    if ax is None:
        figsize = (figsize[0], figsize[1] * len(target_genes))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        if figsize is not None:
            raise ValueError("figsize should not be set if ax is set.")
        fig = ax.get_figure()
        figsize = fig.get_size_inches()
    
    aspect = (figsize[1] / len(target_genes)) / (figsize[0] / expressions.shape[1])
    
    # Determine and apply colormap
    if isinstance(cmap, str):
        vmax = np.quantile(np.abs(expressions).ravel(), 0.95)
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax), 
            cmap=cmap
        )
    
    if hasattr(cmap, "to_rgba"):
        im = ax.imshow(cmap.to_rgba(expressions), aspect=aspect, interpolation='none')
    else:
        im = ax.imshow(expressions, aspect=aspect, interpolation='none', cmap=cmap)
        plt.colorbar(im, label="Expression (Log CPM)")  # Updated label
    
    # Set pseudotime labels as x axis labels
    ax.set_xlabel("Pseudotime")
    num_ticks = 10
    tick_positions = np.linspace(0, expressions.shape[1] - 1, num_ticks, dtype=int)
    tick_labels = dx.iloc[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{x:.6f}" for x in tick_labels], rotation=45, ha="right")
    
    # Set target gene labels
    ax.set_yticks(list(range(len(target_genes))))
    ax.set_yticklabels(target_genes)
    
    # Add grid lines to separate rows
    ax.set_yticks(np.arange(len(target_genes) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    
    return fig, ax, cmap

if __name__ == "__main__":
    data_file = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2/output/dynamic.h5'
    lf_dir = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/other_files/latent_factors'
    output_folder = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2/output'
    dictys_dynamic_object = load_dynamic_object(data_file)
    print('loaded dynamic object')
    # read the 2 txt files in latent_factors_dir
    latent_factors_file_z3 = os.path.join(lf_dir, 'feature_list_Z3.txt')
    latent_factors_file_z11 = os.path.join(lf_dir, 'feature_list_Z11.txt')
    #tab separated files
    latent_factors_df_z3 = pd.read_csv(latent_factors_file_z3, sep='\t')
    latent_factors_df_z11 = pd.read_csv(latent_factors_file_z11, sep='\t')
    latent_factors_genes_z3 = latent_factors_df_z3.iloc[:, 0].tolist()
    latent_factors_genes_z11 = latent_factors_df_z11.iloc[:, 0].tolist()
    #create a set of all the genes
    z11_genes = set(latent_factors_genes_z11)
    z3_genes = set(latent_factors_genes_z3)
    significant_latent_factors_genes = list(z11_genes.union(z3_genes))
    print(len(significant_latent_factors_genes))
    # get the grn edges for the latent factor genes
    tf_target_edges_lf = intersect_grn_edges_with_lf(dictys_dynamic_object, significant_latent_factors_genes)
    print(len(tf_target_edges_lf))
